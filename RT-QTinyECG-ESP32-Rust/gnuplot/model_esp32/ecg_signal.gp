# gnuplot/model_esp32/ecg_signal.gp
# =====================================
# ESP32 int8 model: Raw & Filtered ECG Signal
# Source: data/esp32_predictions.csv  (UART feed evaluator output)
# Output: images/model_esp32/ecg_signal.png
#
# Columns in esp32_predictions.csv:
#   1: index
#   2: time_ms
#   3: adc_raw
#   4: filtered
#   5: gt_label
#   6: pc_prediction
#   7: esp32_prediction
#   8: round_trip_ms
#
# Usage: gnuplot gnuplot/model_esp32/ecg_signal.gp

set terminal pngcairo enhanced font "Arial,11" size 1200,600 background "#1a1a2e"
set output "images/model_esp32/ecg_signal.png"

set style line 1  lc rgb "#4cc9f0" lw 1.5 lt 1    # Raw ADC     — sky blue
set style line 2  lc rgb "#f4a261" lw 2.0 lt 1    # Filtered    — orange
set style line 3  lc rgb "#f72585" lw 1.0 lt 2    # Alerts      — pink
set style line 11 lc rgb "#333355"

set border lc rgb "#aaaacc"
set tics textcolor rgb "#ccccdd"
set key textcolor rgb "#ccccdd" top right box lw 1 lc rgb "#444466"
set grid ls 11

DATA = "data/esp32_predictions.csv"
set datafile separator ","

set title "ESP32 int8 — Raw & Filtered ECG Signal (UART Feed)" \
    textcolor rgb "#e0e0ff" font "Arial Bold,14"
set xlabel "Time (ms)" textcolor rgb "#aaaacc"
set ylabel "ADC Value (0–4095)" textcolor rgb "#aaaacc"
set autoscale x
set yrange [0:4300]

# col 2=time_ms, col 3=adc_raw, col 4=filtered, col 7=esp32_prediction
plot DATA every ::1 using 2:3 with lines ls 1 title "Raw ADC (from PC feed)", \
     DATA every ::1 using 2:4 with lines ls 2 title "Filtered (Moving Avg)", \
     DATA every ::1 using 2:($7 == 1 ? $3 : 1/0) \
         with points pt 7 ps 0.9 lc rgb "#f72585" title "ESP32 Prediction=Abnormal"

set output
print "Saved: images/model_esp32/ecg_signal.png"
