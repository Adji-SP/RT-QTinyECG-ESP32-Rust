# gnuplot/plot_training_history.gp
# ==================================
# Line chart: Model accuracy improvement across fine-tuning iterations.
# Shows old (before) vs new (after) accuracy per run.
# Also marks the baseline (threshold classifier) and target (85%).
#
# Run: gnuplot gnuplot/plot_training_history.gp

set terminal pngcairo size 900,520 enhanced font "Sans,11" background "#1a1a2e"
set output "images/evaluation/training_history.png"

set border lc rgb "#cccccc"
set tics textcolor rgb "#cccccc"
set key textcolor rgb "#cccccc" top left box lc rgb "#444444"
set grid lc rgb "#333355" lt 1 lw 1

set title "Fine-Tuning History: Accuracy Improvement per Run" \
          textcolor rgb "#ffffff" font "Sans Bold,13"
set xlabel "Fine-Tune Iteration" textcolor rgb "#cccccc"
set ylabel "Accuracy"            textcolor rgb "#cccccc"
set yrange [0:1.05]
set ytics 0.1
set xrange [0.5:*]
set xtics 1 textcolor rgb "#cccccc"

set datafile separator ","

# ── Reference lines ───────────────────────────────────────────────────────────
# Threshold classifier baseline (current ~0.578)
set arrow 1 from 0.5, 0.578 to graph 1.02, 0.578 \
    nohead lc rgb "#888888" lw 1 dt 3
set label 1 "Baseline\n(0.578)" at graph 1.02, 0.578 \
    textcolor rgb "#888888" font "Sans,9" left

# Target line at 85%
set arrow 2 from 0.5, 0.85 to graph 1.02, 0.85 \
    nohead lc rgb "#7bed9f" lw 1 dt 2
set label 2 "Target\n(0.85)" at graph 1.02, 0.85 \
    textcolor rgb "#7bed9f" font "Sans,9" left

# Perfect accuracy reference
set arrow 3 from 0.5, 1.00 to graph 1.02, 1.00 \
    nohead lc rgb "#444444" lw 1 dt 3

# ── Style lines ───────────────────────────────────────────────────────────────
set style line 1 lc rgb "#4cc9f0" pt 7 ps 1.2 lw 2.5  # new accuracy — blue
set style line 2 lc rgb "#f72585" pt 5 ps 1.0 lw 1.5 dt 2  # old accuracy — pink dashed
set style line 3 lc rgb "#ffd166" pt 9 ps 0.8 lw 1.0  # trend

# Check if real data (new_accuracy column) or placeholder
# CSV columns: iteration, old_accuracy, new_accuracy, (note)

plot "data/plots/training_history.csv" \
         using 1:3 title "After fine-tune" with linespoints ls 1, \
     "data/plots/training_history.csv" \
         using 1:2 title "Before fine-tune" with linespoints ls 2

print "Saved: images/training_history.png"
