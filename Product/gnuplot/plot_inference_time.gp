# plot_inference_time.gp
# =======================
# GNUPlot script: Plot inference time vs time.
#
# Input:  data/simulated_realtime_log.csv
# Output: images/inference_time.png
#
# Columns in CSV:
#   1: time_ms
#   2: adc_value
#   3: filtered_value
#   4: inference_us    ← this is plotted
#   5: prediction
#   6: alert
#   7: alert_latency_ms
#
# Zero values indicate the ring buffer was not yet full (no inference run).
# Only non-zero values are plotted.
#
# The horizontal red line shows the 4 ms (4000 µs) sampling budget.
# Inference time should stay well below this line for real-time safety.
#
# Usage:
#   gnuplot gnuplot/plot_inference_time.gp

# ── Output settings ──────────────────────────────────────────────────────────
set terminal pngcairo enhanced font "Arial,11" size 1200,500 background "#1a1a2e"
set output "images/inference_time.png"

# ── Color scheme ─────────────────────────────────────────────────────────────
set style line 1 lc rgb "#a8dadc" lw 1.5 lt 1    # Inference time bars
set style line 2 lc rgb "#e63946" lw 2.0 lt 2 dt 2   # Budget limit (dashed red)
set style line 3 lc rgb "#f4a261" lw 1.5 lt 1    # Running average
set style line 11 lc rgb "#333355"

# ── Global styles ────────────────────────────────────────────────────────────
set border lc rgb "#aaaacc"
set tics textcolor rgb "#ccccdd"
set key textcolor rgb "#ccccdd"
set grid ls 11
set key top right box lw 1 lc rgb "#444466"

DATA_FILE = "data/simulated_realtime_log.csv"
set datafile separator ","

BUDGET_US = 4000   # 4 ms in microseconds (sampling interval @ 250 Hz)

# ── Plot ──────────────────────────────────────────────────────────────────────
set title "RT-QTinyECG — Inference Time per Window" \
    textcolor rgb "#e0e0ff" font "Arial Bold,13"
set xlabel "Time (ms)" textcolor rgb "#aaaacc"
set ylabel "Inference Time (µs)" textcolor rgb "#aaaacc"

# Only plot non-zero inference times
# Column 4 = inference_us; skip zeros with conditional
set autoscale x
set yrange [0:*]

plot \
    DATA_FILE every ::1 using 1:($4 > 0 ? $4 : 1/0) with impulses ls 1 title "Inference Time (µs)", \
    BUDGET_US with lines ls 2 title sprintf("Sampling Budget (%d µs = 4 ms)", BUDGET_US)

set output
print "Saved: images/inference_time.png"
