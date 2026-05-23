@echo off
:: ============================================================
:: run_all_plots.bat
:: RT-QTinyECG-ESP32-Rust — Render All GNUPlot Charts
::
:: Organizes output into:
::   images/model_pc/     — PC float32 model charts
::   images/model_esp32/  — ESP32 int8 model charts
::   images/evaluation/   — Cross-model comparison charts
::
:: Usage:  run_all_plots.bat
:: ============================================================

setlocal
set "ROOT=%~dp0"
set "PYTHONIOENCODING=utf-8"
cd /d "%ROOT%"

echo.
echo ============================================================
echo   RT-QTinyECG -- GNUPlot Chart Renderer
echo ============================================================
echo.

:: ── Export all plot data CSVs ─────────────────────────────────────────────────
echo [1/4] Exporting plot data from model outputs...
py python\export_plots_data.py
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: export_plots_data.py failed.
    pause & exit /b 1
)
echo.

:: ── PC float32 model charts ───────────────────────────────────────────────────
echo [2/4] Rendering PC float32 model charts  ^→  images\model_pc\
gnuplot gnuplot\model_pc\ecg_signal.gp     2>nul && echo   OK ecg_signal.png
gnuplot gnuplot\model_pc\alert_latency.gp  2>nul && echo   OK alert_latency.png
gnuplot gnuplot\model_pc\inference_time.gp 2>nul && echo   OK inference_time.png
echo.

:: ── ESP32 int8 model charts ───────────────────────────────────────────────────
echo [3/4] Rendering ESP32 int8 model charts  ^→  images\model_esp32\
gnuplot gnuplot\model_esp32\ecg_signal.gp     2>nul && echo   OK ecg_signal.png
gnuplot gnuplot\model_esp32\alert_latency.gp  2>nul && echo   OK alert_latency.png
gnuplot gnuplot\model_esp32\inference_time.gp 2>nul && echo   OK inference_time.png
echo.

:: ── Evaluation / comparison charts ───────────────────────────────────────────
echo [4/4] Rendering evaluation charts        ^→  images\evaluation\
gnuplot gnuplot\evaluation\accuracy_comparison.gp 2>nul && echo   OK accuracy_comparison.png
gnuplot gnuplot\evaluation\model_weights.gp       2>nul && echo   OK model_weights.png
gnuplot gnuplot\evaluation\quantization_error.gp  2>nul && echo   OK quantization_error.png
gnuplot gnuplot\evaluation\confusion_matrix.gp    2>nul && echo   OK confusion_matrix.png
gnuplot gnuplot\evaluation\training_history.gp    2>nul && echo   OK training_history.png
echo.

:: ── Summary ───────────────────────────────────────────────────────────────────
echo ============================================================
echo   All charts rendered!
echo.
echo   images\model_pc\
echo     ecg_signal.png         Raw + filtered ECG (PC float32)
echo     alert_latency.png      Prediction + alert timeline (PC)
echo     inference_time.png     Inference µs per window (PC)
echo.
echo   images\model_esp32\
echo     ecg_signal.png         Raw + filtered ECG (ESP32 int8)
echo     alert_latency.png      Prediction vs GT + UART latency
echo     inference_time.png     ~25µs int8 vs 4ms budget
echo.
echo   images\evaluation\
echo     accuracy_comparison.png  PC vs ESP32 vs sim metrics
echo     model_weights.png        float32 vs int8 dequant weights
echo     quantization_error.png   Per-weight quant error + distrib
echo     confusion_matrix.png     Side-by-side confusion heatmaps
echo     training_history.png     Fine-tune accuracy over runs
echo ============================================================
echo.
pause
