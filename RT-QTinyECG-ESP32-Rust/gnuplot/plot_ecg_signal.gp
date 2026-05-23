# plot_ecg_signal.gp
# ===================
# GNUPlot script: Plot raw ADC and filtered ECG signal vs time.
#
# Input:  data/simulated_realtime_log.csv
# Output: images/ecg_signal.png
#
# Columns in CSV:
#   1: time_ms
#   2: adc_value
#   3: filtered_value
#   4: inference_us
#   5: prediction
#   6: alert
#   7: alert_latency_ms
#
# Usage:
#   gnuplot gnuplot/plot_ecg_signal.gp

# ── Output settings ──────────────────────────────────────────────────────────
set terminal pngcairo enhanced font "Arial,11" size 1200,600 background "#1a1a2e"
set output "images/ecg_signal.png"

# ── Color scheme (dark theme) ────────────────────────────────────────────────
set style line 1 lc rgb "#00d4ff" lw 1.5 lt 1    # Raw ECG: cyan
set style line 2 lc rgb "#ff6b6b" lw 2.0 lt 1    # Filtered: red-orange
set style line 3 lc rgb "#ffd166" lw 1.0 lt 2    # Alert regions: yellow
set style line 11 lc rgb "#333355"                # Grid lines

# ── Global styles ────────────────────────────────────────────────────────────
set border lc rgb "#aaaacc"
set tics textcolor rgb "#ccccdd"
set key textcolor rgb "#ccccdd"
set grid ls 11
set key top right box lw 1 lc rgb "#444466"

# ── Data file ────────────────────────────────────────────────────────────────
DATA_FILE = "data/simulated_realtime_log.csv"

# ── Title and labels ─────────────────────────────────────────────────────────
set title "RT-QTinyECG — Raw & Filtered ECG Signal" \
    textcolor rgb "#e0e0ff" font "Arial Bold,14"
set xlabel "Time (ms)" textcolor rgb "#aaaacc"
set ylabel "ADC Value (0–4095)" textcolor rgb "#aaaacc"

# ── X range: auto ────────────────────────────────────────────────────────────
set autoscale x
set yrange [0:4300]

# ── Multiplot: show both raw and filtered on the same axes ───────────────────
set datafile separator ","

# Skip header row with 'every ::1' (start from row 2)
plot \
    DATA_FILE every ::1 using 1:2 with lines ls 1 title "Raw ADC (GPIO34)", \
    DATA_FILE every ::1 using 1:3 with lines ls 2 title "Filtered (Moving Avg)", \
    DATA_FILE every ::1 using 1:($6 == 1 ? $2 : 1/0) with points \
        pt 7 ps 0.8 lc rgb "#ff4444" title "Alert Active"

set output
print "Saved: images/ecg_signal.png"
