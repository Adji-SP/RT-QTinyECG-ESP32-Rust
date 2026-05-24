# gnuplot/model_esp32/inference_time.gp
# =====================================
# ESP32 int8 model: Inference Time Estimate per Window
# Source: data/esp32_predictions.csv
# Output: images/model_esp32/inference_time.png
#
# NOTE: When using UART Feed (dry-run), inference_us is not directly
# available from the ESP32 (it would require CCOUNT register reads in
# firmware). This chart shows a "constant" estimate of ~25 µs (int8
# threshold) vs the 4 ms budget, and overlays the UART round-trip
# latency as the dominant observable delay.
#
# Usage: gnuplot gnuplot/model_esp32/inference_time.gp

set terminal pngcairo enhanced font "Arial,11" size 1200,500 background "#1a1a2e"
set output "images/model_esp32/inference_time.png"

set style line 1  lc rgb "#f72585" lw 1.5 lt 1        # ESP32 ~inf time — pink
set style line 2  lc rgb "#e63946" lw 2.0 lt 2 dt 2   # Budget limit    — red
set style line 3  lc rgb "#ffd166" lw 1.5 lt 1        # UART latency    — yellow
set style line 11 lc rgb "#333355"

set border lc rgb "#aaaacc"
set tics textcolor rgb "#ccccdd"
set key textcolor rgb "#ccccdd" top right box lw 1 lc rgb "#444466"
set grid ls 11

DATA = "data/esp32_predictions.csv"
set datafile separator ","

# Estimated int8 inference time: ~25 µs @ 240 MHz
# (threshold classifier), ~60–120 µs (MLP int8)
EST_INF_US = 25
BUDGET_US  = 4000

set title "ESP32 int8 — Estimated Inference Time vs UART Overhead" \
    textcolor rgb "#e0e0ff" font "Arial Bold,13"
set xlabel "Time (ms)" textcolor rgb "#aaaacc"
set ylabel "Time (µs)" textcolor rgb "#aaaacc"
set autoscale x
set yrange [0:*]

# Convert round_trip_ms (col 8) to µs for same axis
# Show UART round-trip in µs and constant int8 inference estimate
plot DATA every ::1 using 2:($8 > 0 ? $8*1000 : 1/0) \
         with impulses ls 3 title "UART Round-Trip (µs, hardware only)", \
     DATA every ::1 using 2:($7 >= 0 ? EST_INF_US : 1/0) \
         with dots lc rgb "#f72585" title sprintf("Est. int8 Inference (~%d µs)", EST_INF_US), \
     BUDGET_US with lines ls 2 \
         title sprintf("4 ms Budget (%d µs)", BUDGET_US)

set output
print "Saved: images/model_esp32/inference_time.png"
