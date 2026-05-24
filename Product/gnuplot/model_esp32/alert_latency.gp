# gnuplot/model_esp32/alert_latency.gp
# =====================================
# ESP32 int8 model: Prediction Timeline & UART Round-Trip Latency
# Source: data/esp32_predictions.csv
# Output: images/model_esp32/alert_latency.png
#
# Panel 1: ESP32 prediction vs Ground Truth vs PC prediction over time
# Panel 2: UART round-trip latency per sample (hardware mode only;
#           all zeros in dry-run)
#
# Usage: gnuplot gnuplot/model_esp32/alert_latency.gp

set terminal pngcairo enhanced font "Arial,11" size 1200,700 background "#1a1a2e"
set output "images/model_esp32/alert_latency.png"

set style line 1  lc rgb "#f72585" lw 2.0 lt 1    # ESP32 pred  — pink
set style line 2  lc rgb "#4cc9f0" lw 1.5 lt 1    # PC pred     — cyan
set style line 3  lc rgb "#7bed9f" lw 1.5 lt 2    # GT label    — green
set style line 4  lc rgb "#ffd166" lw 1.5 lt 1    # Round-trip  — yellow
set style line 11 lc rgb "#333355"

set border lc rgb "#aaaacc"
set tics textcolor rgb "#ccccdd"
set key textcolor rgb "#ccccdd" top right box lw 1 lc rgb "#444466"
set grid ls 11

DATA = "data/esp32_predictions.csv"
set datafile separator ","

# Columns: 1=index, 2=time_ms, 3=adc_raw, 4=filtered,
#          5=gt_label, 6=pc_prediction, 7=esp32_prediction, 8=round_trip_ms

set multiplot layout 2,1 \
    title "ESP32 int8 — Prediction Timeline & UART Latency" \
    textcolor rgb "#e0e0ff" font "Arial Bold,13"

# ── Panel 1: Predictions vs Ground Truth ─────────────────────────────────────
set title "ESP32 Prediction vs PC Prediction vs Ground Truth" \
    textcolor rgb "#ccddff"
set xlabel ""
set ylabel "State (0=Normal, 1=Abnormal)" textcolor rgb "#aaaacc"
set yrange [-0.2:1.7]
set autoscale x

plot DATA every ::1 using 2:($7 >= 0 ? $7 : 1/0) \
         with steps ls 1 lw 2.5 title "ESP32 int8", \
     DATA every ::1 using 2:($6 >= 0 ? $6 : 1/0) \
         with steps ls 2 lw 1.5 title "PC float32", \
     DATA every ::1 using 2:($5 >= 0 ? $5 : 1/0) \
         with steps ls 3 lw 1.0 dt 2 title "Ground Truth"

# ── Panel 2: UART Round-Trip Latency ─────────────────────────────────────────
set title "UART Round-Trip Latency (0 ms = dry-run / simulation)" \
    textcolor rgb "#ccddff"
set xlabel "Time (ms)" textcolor rgb "#aaaacc"
set ylabel "Latency (ms)" textcolor rgb "#aaaacc"
set yrange [0:*]
set autoscale x

# Budget line: 4 ms (250 Hz)
set arrow 1 from graph 0, first 4.0 to graph 1, first 4.0 \
    nohead lc rgb "#e63946" lw 1 dt 2
set label 1 "4 ms budget" at graph 0.82, first 4.2 \
    textcolor rgb "#e63946" font "Arial,9"

plot DATA every ::1 using 2:($8 > 0 ? $8 : 1/0) \
     with impulses ls 4 lw 1.5 title "UART Round-Trip (ms)"

unset arrow 1
unset label 1
unset multiplot
set output
print "Saved: images/model_esp32/alert_latency.png"
