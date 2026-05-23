@echo off
:: ============================================================
:: run_uart_eval.bat
:: RT-QTinyECG-ESP32-Rust -- One-Click UART Evaluation Runner
::
:: Usage:
::   run_uart_eval.bat             -> hardware mode, COM3 (default)
::   run_uart_eval.bat COM4        -> hardware mode, COM4
::   run_uart_eval.bat --dry-run   -> simulate ESP32 on PC (no hardware)
::   run_uart_eval.bat --help      -> show help
:: ============================================================

:: Use plain setlocal (no delayed expansion) to avoid if/else issues
setlocal

set "ROOT=%~dp0"
set "PORT=COM3"
set "DRY_RUN=0"
set "PYTHONIOENCODING=utf-8"

:: ── Parse arguments ───────────────────────────────────────────────────────────
if /I "%~1"=="--help"    goto show_help
if /I "%~1"=="--dry-run" set "DRY_RUN=1"
if /I "%~1"=="COM1"      set "PORT=COM1"
if /I "%~1"=="COM2"      set "PORT=COM2"
if /I "%~1"=="COM3"      set "PORT=COM3"
if /I "%~1"=="COM4"      set "PORT=COM4"
if /I "%~1"=="COM5"      set "PORT=COM5"
if /I "%~1"=="COM6"      set "PORT=COM6"
if /I "%~1"=="COM7"      set "PORT=COM7"
if /I "%~1"=="COM8"      set "PORT=COM8"
if /I "%~2"=="--dry-run" set "DRY_RUN=1"

:: ── Banner ────────────────────────────────────────────────────────────────────
echo.
echo ============================================================
echo   RT-QTinyECG-ESP32-Rust -- UART Feed Evaluation Pipeline
echo ============================================================
if "%DRY_RUN%"=="1" (
    echo   Mode: DRY-RUN -- simulating ESP32 on PC, no hardware
) else (
    echo   Mode: HARDWARE -- ESP32 connected on %PORT%
)
echo   Root: %ROOT%
echo ============================================================
echo.

:: ── Step 0: Toolchain ────────────────────────────────────────────────────────
echo [Step 0/5] Activating ESP32 Rust toolchain...
powershell -Command ". $HOME\export-esp.ps1" >nul 2>&1
echo           Toolchain ready.
echo.

:: ── Hardware vs Dry-Run branch ────────────────────────────────────────────────
if "%DRY_RUN%"=="1" goto skip_hardware

:: ── Step 1: Build firmware (hardware only) ────────────────────────────────────
echo [Step 1/5] Building ESP32 UART-feed firmware...
cd /d "%ROOT%firmware\esp32-rust"
powershell -Command ". $HOME\export-esp.ps1; cargo build --release --features uart-feed 2>&1 | Select-String -Pattern 'error\[|Finished|Compiling ecg'"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Firmware build failed.
    pause & exit /b 1
)
echo           Build OK.
echo.

:: ── Step 2: Flash firmware (hardware only) ───────────────────────────────────
echo [Step 2/5] Flashing firmware to ESP32 on %PORT%...
espflash flash "target\xtensa-esp32-none-elf\release\ecg-esp32" --port %PORT%
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Flash failed. Check cable and port.
    echo TIP:   Run with --dry-run to test without hardware.
    pause & exit /b 1
)
echo           Flash OK. Waiting 3 s for ESP32 to boot...
cd /d "%ROOT%"
timeout /t 3 /nobreak >nul
echo.
goto step3_hardware

:skip_hardware
echo [Step 1/5] SKIPPED -- dry-run, no build needed
echo [Step 2/5] SKIPPED -- dry-run, no flash needed
echo.

:: ── Step 3 (dry-run) ─────────────────────────────────────────────────────────
:step3_dryrun
echo [Step 3/5] Running UART Feed Evaluator (DRY-RUN)...
cd /d "%ROOT%"
py python\uart_feed_evaluator.py --dry-run
if %ERRORLEVEL% NEQ 0 (
    echo ERROR in uart_feed_evaluator.py
    pause & exit /b 1
)
echo.
goto step4

:: ── Step 3 (hardware) ────────────────────────────────────────────────────────
:step3_hardware
echo [Step 3/5] Running UART Feed Evaluator on %PORT%...
cd /d "%ROOT%"
py python\uart_feed_evaluator.py --port %PORT%
if %ERRORLEVEL% NEQ 0 (
    echo ERROR in uart_feed_evaluator.py
    pause & exit /b 1
)
echo.

:: ── Step 4: Compare models ────────────────────────────────────────────────────
:step4
echo [Step 4/5] Comparing PC float32 vs ESP32 int8 predictions...
py python\compare_models.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR in compare_models.py
    pause & exit /b 1
)
echo.

:: ── Step 5: Optimization report ──────────────────────────────────────────────
echo [Step 5/5] Generating optimization report...
py python\optimization_report.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR in optimization_report.py
    pause & exit /b 1
)
echo.

:: ── Bonus: Charts ─────────────────────────────────────────────────────────────────
echo [Bonus] Generating GNUPlot visualizations...
py python\export_plots_data.py >nul 2>&1
gnuplot gnuplot\plot_accuracy_comparison.gp 2>nul && echo   OK: images\accuracy_comparison.png
gnuplot gnuplot\plot_model_weights.gp       2>nul && echo   OK: images\model_weights.png
gnuplot gnuplot\plot_quantization_error.gp  2>nul && echo   OK: images\quantization_error.png
gnuplot gnuplot\plot_confusion_matrix.gp    2>nul && echo   OK: images\confusion_matrix.png
gnuplot gnuplot\plot_training_history.gp    2>nul && echo   OK: images\training_history.png
echo.

:: ── Done ──────────────────────────────────────────────────────────────────────────────
echo ============================================================
echo   Pipeline complete! Output files:
echo     data\esp32_predictions.csv     -- ESP32 predictions
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
echo   PORT        COM port for ESP32  (default: COM3)
echo   --dry-run   No hardware -- simulate ESP32 on PC
echo.
echo Examples:
echo   run_uart_eval.bat                 Hardware, COM3
echo   run_uart_eval.bat COM5            Hardware, COM5
echo   run_uart_eval.bat --dry-run       Simulation only
echo.
