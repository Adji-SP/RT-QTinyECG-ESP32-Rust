# gnuplot/model_pc/inference_time.gp
# =====================================
# PC float32 model: Inference Time per Window
# Source: data/simulated_realtime_log.csv
# Output: images/model_pc/inference_time.png
#
# Usage: gnuplot gnuplot/model_pc/inference_time.gp

set terminal pngcairo enhanced font "Arial,11" size 1200,500 background "#1a1a2e"
set output "images/model_pc/inference_time.png"

set style line 1  lc rgb "#a8dadc" lw 1.5 lt 1        # Inference bars  — teal
set style line 2  lc rgb "#e63946" lw 2.0 lt 2 dt 2   # Budget limit    — red dashed
set style line 11 lc rgb "#333355"

set border lc rgb "#aaaacc"
set tics textcolor rgb "#ccccdd"
set key textcolor rgb "#ccccdd" top right box lw 1 lc rgb "#444466"
set grid ls 11

DATA = "data/simulated_realtime_log.csv"
set datafile separator ","

BUDGET_US = 4000    # 4 ms = 4000 µs (250 Hz sampling budget)

set title "PC float32 — Inference Time per Window" \
    textcolor rgb "#e0e0ff" font "Arial Bold,13"
set xlabel "Time (ms)" textcolor rgb "#aaaacc"
set ylabel "Inference Time (µs)" textcolor rgb "#aaaacc"
set autoscale x
set yrange [0:*]

# Col 4 = inference_us; skip zeros (buffer fill phase)
plot DATA every ::1 using 1:($4 > 0 ? $4 : 1/0) \
         with impulses ls 1 title "Inference Time µs (float32)", \
     BUDGET_US with lines ls 2 \
         title sprintf("4 ms Budget (%d µs)", BUDGET_US)

set output
print "Saved: images/model_pc/inference_time.png"
