# gnuplot/plot_confusion_matrix.gp
# ==================================
# 2x2 confusion matrix heatmap for PC float32 and ESP32 int8 models.
# Uses GNUPlot's image/palette to create a color-coded grid.
#
# Run: gnuplot gnuplot/plot_confusion_matrix.gp

set terminal pngcairo size 1000,500 enhanced font "Sans,11" background "#1a1a2e"
set output "images/evaluation/confusion_matrix.png"

set border lc rgb "#cccccc"
set tics textcolor rgb "#cccccc"
set datafile separator ","

# ── Color palette: dark blue → bright pink/white for high values ──────────────
set palette defined (0 "#1a1a2e", 0.3 "#4361ee", 0.7 "#7209b7", 1 "#f72585")
set cbrange [0:*]
set colorbox

set multiplot layout 1,2 \
    title "Confusion Matrices: PC float32 (left)  vs  ESP32 int8 (right)" \
    textcolor rgb "#ffffff" font "Sans Bold,13"

# Helper function: load confusion matrix as 2x2 image
# CSV format: row, col, row_label, col_label, count

# ── Panel A: PC float32 ───────────────────────────────────────────────────────
set title "PC float32 Model" textcolor rgb "#4cc9f0" font "Sans Bold,12"
set xlabel "Predicted Label" textcolor rgb "#cccccc"
set ylabel "True Label"      textcolor rgb "#cccccc"

set xrange [-0.5:1.5]
set yrange [1.5:-0.5]    # flip y so row 0 (Normal) is on top
set xtics ("Normal" 0, "Abnormal" 1) textcolor rgb "#cccccc"
set ytics ("Normal" 0, "Abnormal" 1) textcolor rgb "#cccccc"
set cbtics textcolor rgb "#cccccc"

# Plot heatmap using image
# col=2(row), col=3(col-idx), col=5(count) in confusion_matrix_pc.csv
# We need: x=col, y=row, color=count
plot "data/plots/confusion_matrix_pc.csv" \
     using 2:1:5 with image notitle, \
     "data/plots/confusion_matrix_pc.csv" \
     using 2:1:(sprintf("%d", $5)) \
     with labels textcolor rgb "#ffffff" font "Sans Bold,14" notitle

# ── Panel B: ESP32 int8 ───────────────────────────────────────────────────────
set title "ESP32 int8 Model" textcolor rgb "#f72585" font "Sans Bold,12"
set xlabel "Predicted Label" textcolor rgb "#cccccc"
set ylabel ""                # no label on right panel

plot "data/plots/confusion_matrix_esp32.csv" \
     using 2:1:5 with image notitle, \
     "data/plots/confusion_matrix_esp32.csv" \
     using 2:1:(sprintf("%d", $5)) \
     with labels textcolor rgb "#ffffff" font "Sans Bold,14" notitle

unset multiplot
print "Saved: images/confusion_matrix.png"
