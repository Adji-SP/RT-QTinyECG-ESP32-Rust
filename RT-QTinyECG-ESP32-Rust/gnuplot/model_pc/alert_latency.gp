# gnuplot/model_pc/alert_latency.gp
# =====================================
# PC float32 model: Prediction & Alert Latency
# Source: data/simulated_realtime_log.csv
# Output: images/model_pc/alert_latency.png
#
# Usage: gnuplot gnuplot/model_pc/alert_latency.gp

set terminal pngcairo enhanced font "Arial,11" size 1200,700 background "#1a1a2e"
set output "images/model_pc/alert_latency.png"

set style line 1  lc rgb "#06d6a0" lw 2.0 lt 1    # Alert ON/OFF  — green
set style line 2  lc rgb "#ef476f" lw 1.5 lt 1    # Latency       — pink-red
set style line 3  lc rgb "#ffd166" lw 1.5 lt 1    # Prediction    — yellow
set style line 11 lc rgb "#333355"

set border lc rgb "#aaaacc"
set tics textcolor rgb "#ccccdd"
set key textcolor rgb "#ccccdd" top right box lw 1 lc rgb "#444466"
set grid ls 11

DATA = "data/simulated_realtime_log.csv"
set datafile separator ","

set multiplot layout 2,1 \
    title "PC float32 — Prediction & Alert Timeline" \
    textcolor rgb "#e0e0ff" font "Arial Bold,13"

set title "Prediction & Alert State" textcolor rgb "#ccddff"
set xlabel ""
set ylabel "State (0=Normal, 1=Abnormal)" textcolor rgb "#aaaacc"
set yrange [-0.1:1.5]
set autoscale x
plot DATA every ::1 using 1:5 with steps ls 3 lw 1.5 title "Prediction (float32)", \
     DATA every ::1 using 1:6 with steps ls 1 lw 2.5 title "Alert ON/OFF"

set title "Alert Latency" textcolor rgb "#ccddff"
set xlabel "Time (ms)" textcolor rgb "#aaaacc"
set ylabel "Latency (ms)" textcolor rgb "#aaaacc"
set yrange [0:*]
set autoscale x
plot DATA every ::1 using 1:($7 > 0 ? $7 : 1/0) \
     with impulses ls 2 lw 2 title "Alert Latency (ms)"

unset multiplot
set output
print "Saved: images/model_pc/alert_latency.png"
