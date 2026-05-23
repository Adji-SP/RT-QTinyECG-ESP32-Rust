# gnuplot/model_pc/ecg_signal.gp
# ================================
# PC float32 model: Raw & Filtered ECG Signal
# Source: data/simulated_realtime_log.csv  (PC simulation output)
# Output: images/model_pc/ecg_signal.png
#
# Usage: gnuplot gnuplot/model_pc/ecg_signal.gp

set terminal pngcairo enhanced font "Arial,11" size 1200,600 background "#1a1a2e"
set output "images/model_pc/ecg_signal.png"

set style line 1  lc rgb "#00d4ff" lw 1.5 lt 1    # Raw ADC    — cyan
set style line 2  lc rgb "#ff6b6b" lw 2.0 lt 1    # Filtered   — coral
set style line 3  lc rgb "#ffd166" lw 1.0 lt 2    # Alerts     — yellow
set style line 11 lc rgb "#333355"                 # Grid

set border lc rgb "#aaaacc"
set tics textcolor rgb "#ccccdd"
set key textcolor rgb "#ccccdd" top right box lw 1 lc rgb "#444466"
set grid ls 11

DATA = "data/simulated_realtime_log.csv"
set datafile separator ","

set title "PC float32 — Raw & Filtered ECG Signal" \
    textcolor rgb "#e0e0ff" font "Arial Bold,14"
set xlabel "Time (ms)" textcolor rgb "#aaaacc"
set ylabel "ADC Value (0–4095)" textcolor rgb "#aaaacc"
set autoscale x
set yrange [0:4300]

plot DATA every ::1 using 1:2 with lines ls 1 title "Raw ADC (PC sim)", \
     DATA every ::1 using 1:3 with lines ls 2 title "Filtered (Moving Avg)", \
     DATA every ::1 using 1:($5 == 1 ? $2 : 1/0) \
         with points pt 7 ps 0.9 lc rgb "#f72585" title "Prediction=Abnormal"

set output
print "Saved: images/model_pc/ecg_signal.png"
