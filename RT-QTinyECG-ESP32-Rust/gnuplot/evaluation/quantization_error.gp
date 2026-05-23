# gnuplot/plot_quantization_error.gp
# =====================================
# Two-panel plot:
#   Top:    Absolute quantization error per weight parameter
#   Bottom: Weight distribution histogram (float32 vs int8 dequantized)
#
# Visualizes the accuracy cost of int8 quantization at the weight level.
#
# Run: gnuplot gnuplot/plot_quantization_error.gp

set terminal pngcairo size 1000,700 enhanced font "Sans,11" background "#1a1a2e"
set output "images/evaluation/quantization_error.png"

set border lc rgb "#cccccc"
set tics textcolor rgb "#cccccc"
set key textcolor rgb "#cccccc" top right box lc rgb "#444444"
set grid lc rgb "#333355" lt 1 lw 1
set datafile separator ","

set multiplot layout 2,1 \
    title "Quantization Analysis: float32 → int8" \
    textcolor rgb "#ffffff" font "Sans Bold,13"

# ── Panel A: Absolute error per parameter ─────────────────────────────────────
set title "Absolute Quantization Error per Weight Parameter" \
          textcolor rgb "#ffcc88" font "Sans,11"
set xlabel "Parameter Index" textcolor rgb "#cccccc"
set ylabel "Abs Error"       textcolor rgb "#cccccc"
set yrange [0:*]
set xrange [-1:*]

# Color bars by error magnitude using a gradient (impulse style)
set style fill solid 0.9
set boxwidth 0.6

plot "data/plots/quantization_error.csv" \
     using 1:4 title "float32 - int8 dequant" \
     with impulses lc rgb "#f72585" lw 1.5

# ── Panel B: Weight distribution histogram ────────────────────────────────────
set title "Weight Value Distribution (float32 vs int8 dequantized)" \
          textcolor rgb "#88ccff" font "Sans,11"
set xlabel "Weight Value"   textcolor rgb "#cccccc"
set ylabel "Count"          textcolor rgb "#cccccc"
set yrange [0:*]
set xrange [*:*]
set style data histogram
set style histogram clustered gap 0.5
set style fill solid 0.75 border lc rgb "#000000"
set boxwidth 0.8

plot "data/plots/weight_distribution.csv" \
     using 2:xtic(1) title "float32"       lc rgb "#4cc9f0", \
     "data/plots/weight_distribution.csv" \
     using 3          title "int8 dequant" lc rgb "#f72585"

unset multiplot
print "Saved: images/quantization_error.png"
