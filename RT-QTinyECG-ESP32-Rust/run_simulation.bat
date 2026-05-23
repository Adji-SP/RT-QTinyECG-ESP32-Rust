@echo off
:: ============================================================
:: run_simulation.bat
:: RT-QTinyECG-ESP32-Rust -- PC-Only Simulation Pipeline
::
:: Runs the full analysis pipeline on PC without any ESP32.
:: Useful for: development, testing, demo, and report generation.
::
:: Usage:
::   run_simulation.bat           -- full pipeline
::   run_simulation.bat --quick   -- skip data gen + training if data exists
::   run_simulation.bat --plots   -- only regenerate charts (skip simulation)
:: ============================================================

setlocal
set "ROOT=%~dp0"
set "PYTHONIOENCODING=utf-8"
set "QUICK=0"
set "PLOTS_ONLY=0"

if /I "%~1"=="--quick"   set "QUICK=1"
if /I "%~1"=="--plots"   set "PLOTS_ONLY=1"
if /I "%~2"=="--quick"   set "QUICK=1"

echo.
echo ============================================================
echo   RT-QTinyECG-ESP32-Rust -- PC Simulation Pipeline
echo   (No ESP32 hardware required)
echo ============================================================
echo.

cd /d "%ROOT%"

if "%PLOTS_ONLY%"=="1" goto :charts

:: ── Step 1: Generate data ─────────────────────────────────────────────────────
if "%QUICK%"=="1" (
    if exist "data\sample_ecg.csv" (
        echo [1/7] SKIPPED -- sample_ecg.csv already exists ^(--quick^)
        goto step2
    )
)
echo [1/7] Generating synthetic ECG dataset...
py python\generate_dummy_ecg.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in step 1 & pause & exit /b 1 )

:step2
echo.

:: ── Step 2: Train model ───────────────────────────────────────────────────────
if "%QUICK%"=="1" (
    if exist "data\model.pkl" (
        echo [2/7] SKIPPED -- model.pkl already exists ^(--quick^)
        goto step3
    )
)
echo [2/7] Training PC float32 model + quantizing to int8...
py python\train_simple_model.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in step 2 & pause & exit /b 1 )
py python\quantize_weights.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR: quantize_weights.py failed & pause & exit /b 1 )

:step3
echo.

:: ── Step 3: Realtime simulation ───────────────────────────────────────────────
echo [3/7] Running realtime ECG simulation ^(PC float32^)...
py python\realtime_ecg_simulator.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in step 3 & pause & exit /b 1 )
echo.

:: ── Step 4: Metrics ───────────────────────────────────────────────────────────
echo [4/7] Evaluating PC simulation metrics...
py python\metrics.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in step 4 & pause & exit /b 1 )
echo.

:: ── Step 5: UART Feed Evaluator (dry-run) ────────────────────────────────────
echo [5/7] Running UART Feed Evaluator ^(dry-run — simulated ESP32^)...
py python\uart_feed_evaluator.py --dry-run
if %ERRORLEVEL% NEQ 0 ( echo ERROR in step 5 & pause & exit /b 1 )
echo.

:: ── Step 6: Compare + Optimize ───────────────────────────────────────────────
echo [6/7] Comparing models + generating optimization report...
py python\compare_models.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in compare_models.py & pause & exit /b 1 )
py python\optimization_report.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in optimization_report.py & pause & exit /b 1 )
echo.

:: ── Step 7: Fine-tuning (optional — prompts user) ────────────────────────────
echo [7/7] Fine-tuning on disagreement samples...
py python\fine_tune_model.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: fine_tune_model.py returned error ^(may need more disagreement samples^)
)
echo.

:charts
:: ── Charts ────────────────────────────────────────────────────────────────────
echo ============================================================
echo   Rendering GNUPlot charts...
echo ============================================================
py python\export_plots_data.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR: export_plots_data.py failed & pause & exit /b 1 )
echo.

echo   [PC float32]  images\model_pc\
gnuplot gnuplot\model_pc\ecg_signal.gp     2>nul && echo     OK ecg_signal.png
gnuplot gnuplot\model_pc\alert_latency.gp  2>nul && echo     OK alert_latency.png
gnuplot gnuplot\model_pc\inference_time.gp 2>nul && echo     OK inference_time.png
echo.

echo   [ESP32 int8]  images\model_esp32\
gnuplot gnuplot\model_esp32\ecg_signal.gp     2>nul && echo     OK ecg_signal.png
gnuplot gnuplot\model_esp32\alert_latency.gp  2>nul && echo     OK alert_latency.png
gnuplot gnuplot\model_esp32\inference_time.gp 2>nul && echo     OK inference_time.png
echo.

echo   [Evaluation]  images\evaluation\
gnuplot gnuplot\evaluation\accuracy_comparison.gp 2>nul && echo     OK accuracy_comparison.png
gnuplot gnuplot\evaluation\model_weights.gp       2>nul && echo     OK model_weights.png
gnuplot gnuplot\evaluation\quantization_error.gp  2>nul && echo     OK quantization_error.png
gnuplot gnuplot\evaluation\confusion_matrix.gp    2>nul && echo     OK confusion_matrix.png
gnuplot gnuplot\evaluation\training_history.gp    2>nul && echo     OK training_history.png
echo.

:: ── Summary ───────────────────────────────────────────────────────────────────
echo ============================================================
echo   Pipeline complete!
echo.
echo   Data outputs:
echo     data\simulated_realtime_log.csv   - PC simulation log
echo     data\esp32_predictions.csv        - Simulated ESP32
echo     data\comparison_report.csv        - Model comparison
echo     data\optimization_targets.json    - Optimization priorities
echo.
echo   Charts:
echo     images\model_pc\    - 3 charts: ECG, alert, inference
echo     images\model_esp32\ - 3 charts: ECG, alert, inference
echo     images\evaluation\  - 5 charts: accuracy, weights, quant, cm, history
echo.
echo   Next steps:
echo     run_uart_eval.bat COM3    - evaluate on real ESP32
echo     run_all_plots.bat         - regenerate all charts only
echo ============================================================
echo.
pause
