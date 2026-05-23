# gnuplot/plot_accuracy_comparison.gp
# =====================================
# Clustered bar chart: Accuracy, Precision, Recall, F1
# comparing PC float32 vs ESP32 int8 vs PC simulation
#
# Reads: data/plots/accuracy_chart.csv
# Cols:  metric, PC_float32_feed, ESP32_int8, PC_sim_float32
#
# Run: gnuplot gnuplot/plot_accuracy_comparison.gp

set terminal pngcairo size 900,560 enhanced font "Sans,11" background "#1a1a2e"
set output "images/evaluation/accuracy_comparison.png"

set border lc rgb "#cccccc"
set tics textcolor rgb "#cccccc"
set key textcolor rgb "#cccccc" top right box lc rgb "#555555"
set grid y lc rgb "#333355" lt 1 lw 1
set border 3
set tics nomirror

set title "Model Comparison: Accuracy / Precision / Recall / F1" \
          textcolor rgb "#ffffff" font "Sans Bold,13"
set xlabel "Metric"  textcolor rgb "#cccccc"
set ylabel "Score"   textcolor rgb "#cccccc"
set yrange [0:1.2]
set ytics 0.1

set style data histogram
set style histogram clustered gap 1.5
set style fill solid 0.85 border lc rgb "#111111"
set boxwidth 0.85

set datafile separator ","

# accuracy_chart.csv columns:
# 1=metric(str), 2=PC_float32_feed, 3=ESP32_int8, 4=PC_sim_float32

plot "data/plots/accuracy_chart.csv" \
         using 2:xtic(1) title "PC float32 (feed)"  lc rgb "#4cc9f0", \
     "" using 3           title "ESP32 int8"          lc rgb "#f72585", \
     "" using 4           title "PC float32 (sim)"    lc rgb "#7bed9f"

print "Saved: images/accuracy_comparison.png"
