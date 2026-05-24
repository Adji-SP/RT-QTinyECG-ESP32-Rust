# gnuplot/plot_model_weights.gp
# ================================
# Scatter/lollipop plot: float32 vs dequantized int8 weights
# for Layer 1 (W1) and Layer 2 (W2) of the MLP classifier.
#
# Shows visually how close the int8 approximation is to the original.
#
# Run: gnuplot gnuplot/plot_model_weights.gp

set terminal pngcairo size 1000,700 enhanced font "Sans,11" background "#1a1a2e"
set output "images/evaluation/model_weights.png"

# ── Styles ────────────────────────────────────────────────────────────────────
set border lc rgb "#cccccc"
set tics textcolor rgb "#cccccc"
set key textcolor rgb "#cccccc" top right box lc rgb "#444444"
set grid lc rgb "#333355" lt 1 lw 1

set datafile separator ","

# ── Multiplot: W1 (top) and W2 (bottom) ──────────────────────────────────────
set multiplot layout 2,1 title \
    "MLP Weight Comparison: float32 vs int8 Dequantized" \
    textcolor rgb "#ffffff" font "Sans Bold,13"

# ── Layer 1 weights ───────────────────────────────────────────────────────────
set title "Layer 1 (W1): Input→Hidden  [5 × hidden_size params]" \
          textcolor rgb "#aaaaff" font "Sans,11"
set xlabel "Parameter Index" textcolor rgb "#cccccc"
set ylabel "Weight Value"    textcolor rgb "#cccccc"
set yrange [*:*]
set xrange [-1:*]

set style line 1 lc rgb "#4cc9f0" pt 7 ps 0.8 lw 1.5  # float32 — blue circles
set style line 2 lc rgb "#f72585" pt 5 ps 0.8 lw 1.5  # int8    — pink squares
set style line 3 lc rgb "#555577" lw 1 lt 1            # zero line

# Zero reference line
set arrow from -1, 0 to graph 1.05, 0 nohead lc rgb "#555577" lw 1 dt 2

plot "data/plots/model_weights_w1.csv" \
         using 1:2 title "float32" with linespoints ls 1, \
     "data/plots/model_weights_w1.csv" \
         using 1:3 title "int8 dequant" with linespoints ls 2

# ── Layer 2 weights ───────────────────────────────────────────────────────────
set title "Layer 2 (W2): Hidden→Output  [hidden_size × 1 params]" \
          textcolor rgb "#aaffaa" font "Sans,11"

set arrow from -1, 0 to graph 1.05, 0 nohead lc rgb "#555577" lw 1 dt 2

plot "data/plots/model_weights_w2.csv" \
         using 1:2 title "float32" with linespoints ls 1, \
     "data/plots/model_weights_w2.csv" \
         using 1:3 title "int8 dequant" with linespoints ls 2

unset multiplot
print "Saved: images/model_weights.png"
