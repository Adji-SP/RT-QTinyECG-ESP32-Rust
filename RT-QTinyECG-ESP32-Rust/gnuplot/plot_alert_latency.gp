# plot_alert_latency.gp
# ======================
# GNUPlot script: Plot alert status and alert latency vs time.
#
# Input:  data/simulated_realtime_log.csv
# Output: images/alert_latency.png
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
#   gnuplot gnuplot/plot_alert_latency.gp

# ── Output settings ──────────────────────────────────────────────────────────
set terminal pngcairo enhanced font "Arial,11" size 1200,700 background "#1a1a2e"
set output "images/alert_latency.png"

# ── Color scheme ─────────────────────────────────────────────────────────────
set style line 1 lc rgb "#06d6a0" lw 2.0 lt 1    # Alert state: green
set style line 2 lc rgb "#ef476f" lw 1.5 lt 1    # Latency: red/pink
set style line 3 lc rgb "#ffd166" lw 1.5 lt 1    # Prediction: yellow
set style line 11 lc rgb "#333355"

# ── Global styles ────────────────────────────────────────────────────────────
set border lc rgb "#aaaacc"
set tics textcolor rgb "#ccccdd"
set key textcolor rgb "#ccccdd"
set grid ls 11
set key top right box lw 1 lc rgb "#444466"

DATA_FILE = "data/simulated_realtime_log.csv"
set datafile separator ","

# ── Multiplot: 2 rows ────────────────────────────────────────────────────────
set multiplot layout 2,1 title "RT-QTinyECG — Alert Status & Latency" \
    textcolor rgb "#e0e0ff" font "Arial Bold,13"

# ── Plot 1: Alert state (0/1) and Prediction (0/1) ──────────────────────────
set title "Prediction & Alert State" textcolor rgb "#ccddff"
set xlabel ""
set ylabel "State (0=Normal, 1=Abnormal)" textcolor rgb "#aaaacc"
set yrange [-0.1:1.5]
set autoscale x

plot \
    DATA_FILE every ::1 using 1:5 with steps ls 3 lw 1.5 title "Prediction (0/1)", \
    DATA_FILE every ::1 using 1:6 with steps ls 1 lw 2.5 title "Alert ON/OFF"

# ── Plot 2: Alert latency ─────────────────────────────────────────────────────
set title "Alert Latency" textcolor rgb "#ccddff"
set xlabel "Time (ms)" textcolor rgb "#aaaacc"
set ylabel "Latency (ms)" textcolor rgb "#aaaacc"
set yrange [0:*]
set autoscale x

plot \
    DATA_FILE every ::1 using 1:($7 > 0 ? $7 : 1/0) with impulses ls 2 lw 2 title "Alert Latency (ms)"

unset multiplot
set output
print "Saved: images/alert_latency.png"
