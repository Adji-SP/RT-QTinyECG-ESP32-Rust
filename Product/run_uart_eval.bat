@echo off
:: ============================================================
:: run_uart_eval.bat
:: RT-QTinyECG-ESP32-Rust -- One-Click UART Evaluation Runner
::
:: ESP32-S3 version
::
:: Usage:
::   run_uart_eval.bat              -> hardware mode, COM16 default
::   run_uart_eval.bat COM4         -> hardware mode, COM4
::   run_uart_eval.bat COM16        -> hardware mode, COM16
::   run_uart_eval.bat --dry-run    -> simulate ESP32-S3 on PC, no hardware
::   run_uart_eval.bat --help       -> show help
:: ============================================================

setlocal EnableExtensions

set "ROOT=%~dp0"
set "PORT=COM16"
set "DRY_RUN=0"
set "PYTHONIOENCODING=utf-8"

:: IMPORTANT:
:: Use a separate Cargo target directory to avoid Windows file-lock errors
:: inside the project target folder.
set "CARGO_TARGET_DIR=C:\rust-target\ecg-esp32s3"
set "CARGO_INCREMENTAL=0"

:: ESP32-S3 target triple
set "ESP_TARGET=xtensa-esp32s3-none-elf"

:: ── Parse arguments ───────────────────────────────────────────────────────────
:parse_args
if "%~1"=="" goto args_done

if /I "%~1"=="--help" goto show_help

if /I "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
    goto parse_args
)

echo %~1 | findstr /R /I "^COM[0-9][0-9]*$" >nul
if %ERRORLEVEL% EQU 0 (
    set "PORT=%~1"
    shift
    goto parse_args
)

echo.
echo ERROR: Unknown argument "%~1"
echo.
goto show_help

:args_done

:: ── Banner ────────────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo   RT-QTinyECG-ESP32-Rust -- UART Feed Evaluation Pipeline
echo   Target: ESP32-S3 / %ESP_TARGET%
echo ============================================================
if "%DRY_RUN%"=="1" (
    echo   Mode: DRY-RUN -- simulating ESP32-S3 on PC, no hardware
) else (
    echo   Mode: HARDWARE -- ESP32-S3 connected on %PORT%
)
echo   Root:             %ROOT%
echo   Cargo target dir: %CARGO_TARGET_DIR%
echo ============================================================
echo.

:: ── Step 0: Toolchain check ───────────────────────────────────────────────────
echo [Step 0/5] Activating ESP Rust toolchain...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$env:CARGO_TARGET_DIR='%CARGO_TARGET_DIR%'; $env:CARGO_INCREMENTAL='0'; . $HOME\export-esp.ps1" >nul 2>&1

if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Could not run export-esp.ps1 in test shell.
    echo          Build step will still try to activate it again.
)

if not exist "%CARGO_TARGET_DIR%" (
    mkdir "%CARGO_TARGET_DIR%" >nul 2>&1
)

echo           Toolchain step done.
echo.

:: ── Hardware vs Dry-Run branch ────────────────────────────────────────────────
if "%DRY_RUN%"=="1" goto skip_hardware

:: ── Step 1: Build firmware, ESP32-S3 target ───────────────────────────────────
echo [Step 1/5] Building ESP32-S3 UART-feed firmware...
cd /d "%ROOT%firmware\esp32-rust"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$env:CARGO_TARGET_DIR='%CARGO_TARGET_DIR%'; $env:CARGO_INCREMENTAL='0'; . $HOME\export-esp.ps1; cargo build --release --target %ESP_TARGET% --features uart-feed -j 1; exit $LASTEXITCODE"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Firmware build failed.
    echo.
    echo If this is still an os error 32 file-lock issue:
    echo   1. Close VS Code completely
    echo   2. Close espflash monitor / serial monitor / PuTTY
    echo   3. Open PowerShell as Administrator
    echo   4. Run:
    echo      Get-Process cargo,rustc,rust-analyzer,espflash,code -ErrorAction SilentlyContinue ^| Stop-Process -Force
    echo.
    pause
    exit /b 1
)

echo           Build OK.
echo.

:: ── Step 2: Locate and flash firmware ─────────────────────────────────────────
echo [Step 2/5] Locating firmware output...

set "FIRMWARE=%CARGO_TARGET_DIR%\%ESP_TARGET%\release\ecg-esp32"

echo           Firmware: %FIRMWARE%
echo.

if not exist "%FIRMWARE%" (
    echo ERROR: Firmware file not found:
    echo        %FIRMWARE%
    echo.
    echo Try running this manually:
    echo   cd /d "%ROOT%firmware\esp32-rust"
    echo   set CARGO_TARGET_DIR=%CARGO_TARGET_DIR%
    echo   set CARGO_INCREMENTAL=0
    echo   cargo build --release --target %ESP_TARGET% --features uart-feed -j 1
    echo.
    pause
    exit /b 1
)

echo [Step 2/5] Flashing firmware to ESP32-S3 on %PORT%...
espflash flash "%FIRMWARE%" --port %PORT%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Flash failed. Check cable, boot mode, and port.
    echo TIP:   Run with --dry-run to test without hardware.
    pause
    exit /b 1
)

echo           Flash OK. Waiting 3 s for ESP32-S3 to boot...
cd /d "%ROOT%"
timeout /t 3 /nobreak >nul
echo.
goto step3_hardware

:skip_hardware
echo [Step 1/5] SKIPPED -- dry-run, no build needed
echo [Step 2/5] SKIPPED -- dry-run, no flash needed
echo.

:: ── Step 3 dry-run ────────────────────────────────────────────────────────────
:step3_dryrun
echo [Step 3/5] Running UART Feed Evaluator, DRY-RUN...
cd /d "%ROOT%"

py python\uart_feed_evaluator.py --dry-run

if %ERRORLEVEL% NEQ 0 (
    echo ERROR in uart_feed_evaluator.py
    pause
    exit /b 1
)

echo.
goto step4

:: ── Step 3 hardware ───────────────────────────────────────────────────────────
:step3_hardware
echo [Step 3/5] Running UART Feed Evaluator on %PORT%...
cd /d "%ROOT%"

py python\uart_feed_evaluator.py --port %PORT%

if %ERRORLEVEL% NEQ 0 (
    echo ERROR in uart_feed_evaluator.py
    pause
    exit /b 1
)

echo.

:: ── Step 4: Compare models ────────────────────────────────────────────────────
:step4
echo [Step 4/5] Comparing PC float32 vs ESP32-S3 int8 predictions...

py python\compare_models.py

if %ERRORLEVEL% NEQ 0 (
    echo ERROR in compare_models.py
    pause
    exit /b 1
)

echo.

:: ── Step 5: Optimization report ───────────────────────────────────────────────
echo [Step 5/5] Generating optimization report...

py python\optimization_report.py

if %ERRORLEVEL% NEQ 0 (
    echo ERROR in optimization_report.py
    pause
    exit /b 1
)

echo.

:: ── Bonus: GNUPlot charts ─────────────────────────────────────────────────────
echo [Bonus] Generating GNUPlot visualizations...

py python\export_plots_data.py >nul 2>&1

gnuplot gnuplot\plot_accuracy_comparison.gp 2>nul && echo   OK: images\accuracy_comparison.png
gnuplot gnuplot\plot_model_weights.gp       2>nul && echo   OK: images\model_weights.png
gnuplot gnuplot\plot_quantization_error.gp  2>nul && echo   OK: images\quantization_error.png
gnuplot gnuplot\plot_confusion_matrix.gp    2>nul && echo   OK: images\confusion_matrix.png
gnuplot gnuplot\plot_training_history.gp    2>nul && echo   OK: images\training_history.png

echo.

:: ── Done ──────────────────────────────────────────────────────────────────────
echo ============================================================
echo   Pipeline complete! Output files:
echo     data\esp32_predictions.csv     -- ESP32-S3 predictions
echo     data\comparison_report.csv     -- Model comparison
echo     data\optimization_targets.json -- Optimization priorities
echo     images\accuracy_comparison.png -- Accuracy bar chart
echo     images\model_weights.png       -- Weight comparison
echo     images\confusion_matrix.png    -- Confusion matrices
echo     images\quantization_error.png  -- Quantization analysis
echo     images\training_history.png    -- Fine-tune history
echo.
echo   To fine-tune:  py python\fine_tune_model.py
echo   Re-run:        run_uart_eval.bat %PORT%
echo ============================================================
echo.

pause
goto :eof

:show_help
echo.
echo Usage:  run_uart_eval.bat [PORT] [--dry-run]
echo.
echo   PORT        COM port for ESP32-S3, default: COM16
echo   --dry-run   No hardware -- simulate ESP32-S3 on PC
echo.
echo Examples:
echo   run_uart_eval.bat                 Hardware, COM16
echo   run_uart_eval.bat COM5            Hardware, COM5
echo   run_uart_eval.bat COM16           Hardware, COM16
echo   run_uart_eval.bat --dry-run       Simulation only
echo.
pause
goto :eof