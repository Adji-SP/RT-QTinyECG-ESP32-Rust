# Project Documentation Index

## RT-QTinyECG-ESP32-Rust

**Real-Time Quantized ECG Detection on ESP32 Using Embedded Rust**
*Educational embedded systems prototype — not a clinical device*

---

## Quick Navigation

| Document | What It Covers |
|----------|----------------|
| [block_diagram.md](block_diagram.md) | Hardware and software block diagram |
| [wiring_esp32_ad8232.md](wiring_esp32_ad8232.md) | Physical wiring (AD8232 → ESP32) |
| [flowchart.md](flowchart.md) | Firmware execution flowchart |
| [algorithm.md](algorithm.md) | ECG detection algorithm details |
| [real_time_design.md](real_time_design.md) | Real-time constraints and design |
| [evaluation_metrics.md](evaluation_metrics.md) | Metrics definitions and targets |
| [**uart_feed_evaluation.md**](uart_feed_evaluation.md) | **UART Feed: PC→ESP32 validation pipeline** |
| [**optimization_guide.md**](optimization_guide.md) | **TinyML optimization strategies** |

---

## Project Scope

### What This Project Does

```
[Simulated ECG Data]
        │
        ├── Path A: PC float32 model (sklearn MLP)
        │         → baseline accuracy reference
        │
        └── Path B: ESP32 int8 model (Rust firmware)
                  ← fed via UART from PC (no sensor needed)
                  → predictions captured, compared vs Path A
                  → quantization accuracy delta measured
                  → optimization applied, loop repeated
```

### Core Research Question

> *How much accuracy is lost when a float32 MLP classifier is
> quantized to int8 and deployed on an ESP32 microcontroller,
> and what optimizations recover that loss?*

---

## Workflow Summary

```
Phase 1 — Simulation (no hardware)
──────────────────────────────────
generate_dummy_ecg.py   → data/sample_ecg.csv
train_simple_model.py   → data/model.pkl
realtime_ecg_simulator.py → data/simulated_realtime_log.csv
metrics.py              → terminal report
gnuplot *.gp            → images/*.png

Phase 2 — Firmware Build
──────────────────────────────────
cd firmware/esp32-rust
cargo build --release   → target/.../ecg-esp32 (67 KB)
espflash flash ...      → ESP32 flashed

Phase 3 — UART Feed Evaluation  [main contribution]
──────────────────────────────────
uart_feed_evaluator.py  → sends ECG to ESP32, captures predictions
compare_models.py       → PC float32 vs ESP32 int8 accuracy delta
optimization_report.py  → identifies next optimization target

Phase 4 — Optimization Loop
──────────────────────────────────
[adjust threshold / architecture / quantization bits]
→ rebuild → reflash → re-evaluate → compare delta
→ repeat until target accuracy met
```

---

## File Structure

```
RT-QTinyECG-ESP32-Rust/
├── README.md
├── requirements.txt
├── run_simulation.bat           ← PC-only full pipeline (no hardware)
├── run_uart_eval.bat            ← Hardware/dry-run UART pipeline
├── run_all_plots.bat            ← Render all 11 GNUPlot charts
│
├── data/
│   ├── sample_ecg.csv           ← Synthetic ECG + ground truth
│   ├── simulated_realtime_log.csv ← PC simulation log (250 Hz)
│   ├── esp32_predictions.csv    ← ESP32 UART feed results
│   ├── comparison_report.csv    ← Side-by-side model comparison
│   ├── optimization_targets.json← Prioritized optimization recs
│   ├── model.pkl / scaler.pkl   ← Trained sklearn MLP
│   ├── quantized_weights.npz    ← int8 weights + scale factors
│   └── plots/                   ← Pre-computed CSVs for GNUPlot
│       ├── accuracy_chart.csv
│       ├── model_weights_w1.csv / w2.csv
│       ├── quantization_error.csv
│       ├── weight_distribution.csv
│       ├── confusion_matrix_pc.csv / esp32.csv
│       └── training_history.csv
│
├── python/
│   ├── generate_dummy_ecg.py
│   ├── preprocessing.py
│   ├── train_simple_model.py
│   ├── realtime_ecg_simulator.py
│   ├── quantize_weights.py
│   ├── export_rust_weights.py
│   ├── metrics.py
│   ├── export_plots_data.py     ← Exports all GNUPlot data CSVs  [NEW]
│   ├── uart_feed_evaluator.py   ← Feeds ESP32 via UART           [NEW]
│   ├── compare_models.py        ← PC vs ESP32 comparison          [NEW]
│   ├── optimization_report.py   ← Prioritized recommendations     [NEW]
│   └── fine_tune_model.py       ← Retrain on disagreement samples [NEW]
│
├── gnuplot/
│   ├── model_pc/                ← PC float32 model charts
│   │   ├── ecg_signal.gp        → images/model_pc/ecg_signal.png
│   │   ├── alert_latency.gp     → images/model_pc/alert_latency.png
│   │   └── inference_time.gp    → images/model_pc/inference_time.png
│   │
│   ├── model_esp32/             ← ESP32 int8 model charts
│   │   ├── ecg_signal.gp        → images/model_esp32/ecg_signal.png
│   │   ├── alert_latency.gp     → images/model_esp32/alert_latency.png
│   │   └── inference_time.gp    → images/model_esp32/inference_time.png
│   │
│   └── evaluation/              ← Cross-model comparison charts
│       ├── accuracy_comparison.gp → images/evaluation/accuracy_comparison.png
│       ├── model_weights.gp       → images/evaluation/model_weights.png
│       ├── quantization_error.gp  → images/evaluation/quantization_error.png
│       ├── confusion_matrix.gp    → images/evaluation/confusion_matrix.png
│       └── training_history.gp    → images/evaluation/training_history.png
│
├── images/
│   ├── model_pc/                ← PC float32 output charts
│   │   ├── ecg_signal.png
│   │   ├── alert_latency.png
│   │   └── inference_time.png
│   ├── model_esp32/             ← ESP32 int8 output charts
│   │   ├── ecg_signal.png
│   │   ├── alert_latency.png
│   │   └── inference_time.png
│   └── evaluation/              ← Cross-model comparison charts
│       ├── accuracy_comparison.png
│       ├── model_weights.png
│       ├── confusion_matrix.png
│       ├── quantization_error.png
│       └── training_history.png
│
├── firmware/esp32-rust/
│   ├── Cargo.toml               ← esp-hal 1.1, features: uart-feed
│   ├── rust-toolchain.toml
│   ├── .cargo/config.toml
│   └── src/
│       ├── main.rs              ← Dual-mode (ADC / UART feed)
│       ├── inference.rs
│       ├── model_weights.rs
│       ├── ring_buffer.rs
│       ├── filter.rs
│       ├── logger.rs
│       ├── uart_feed.rs         ← UART RX via MMIO              [NEW]
│       └── alert.rs
│
└── docs/
    ├── INDEX.md                 ← This file
    ├── block_diagram.md
    ├── wiring_esp32_ad8232.md
    ├── flowchart.md
    ├── algorithm.md
    ├── real_time_design.md
    ├── evaluation_metrics.md
    ├── uart_feed_evaluation.md  ← UART pipeline steps
    └── optimization_guide.md   ← TinyML optimization reference
```

---

## Chart Gallery

### PC float32 Model (`images/model_pc/`)

| Chart | What to Read |
|---|---|
| `ecg_signal.png` | Raw ADC vs filtered signal; pink dots = predicted abnormal |
| `alert_latency.png` | Prediction state over time; alert ON/OFF timeline |
| `inference_time.png` | Per-window inference µs vs 4 ms budget |

### ESP32 int8 Model (`images/model_esp32/`)

| Chart | What to Read |
|---|---|
| `ecg_signal.png` | UART-fed signal; ESP32 prediction markers |
| `alert_latency.png` | ESP32 vs PC vs Ground Truth predictions; UART round-trip |
| `inference_time.png` | ~25 µs int8 estimate vs 4 ms budget vs UART overhead |

### Evaluation / Comparison (`images/evaluation/`)

| Chart | What to Read |
|---|---|
| `accuracy_comparison.png` | Accuracy/Precision/Recall/F1 — 3 model bar chart |
| `model_weights.png` | float32 vs int8 dequant weights per layer |
| `quantization_error.png` | Per-weight error + weight distribution overlay |
| `confusion_matrix.png` | Side-by-side 2×2 heatmaps PC vs ESP32 |
| `training_history.png` | Accuracy before/after each fine-tune run |

---

## Current Status

| Component | Status |
|-----------|--------|
| Python simulation pipeline | ✅ Working |
| ESP32 firmware (ADC mode) | ✅ Builds clean (67,600 bytes) |
| ESP32 firmware (uart-feed) | ✅ Builds clean (`--features uart-feed`) |
| UART Feed Evaluator | ✅ Working (dry-run tested) |
| Cross-model comparison | ✅ Working |
| Optimization report | ✅ Working |
| Fine-tune script | ✅ Created |
| GNUPlot — PC model charts | ✅ 3 charts in `images/model_pc/` |
| GNUPlot — ESP32 model charts | ✅ 3 charts in `images/model_esp32/` |
| GNUPlot — Evaluation charts | ✅ 5 charts in `images/evaluation/` |

---

## Current Metrics (Simulation Baseline)

| Metric | PC float32 (sim) | PC float32 (feed) | ESP32 int8 |
|--------|-----------------|-------------------|------------|
| Accuracy | **95.3%** | 57.9% | 57.9% |
| Precision | 1.000 | 0.000 | 0.000 |
| Recall | 0.883 | 0.000 | 0.000 |
| F1-Score | 0.938 | 0.000 | 0.000 |
| Model size | 228 bytes | 228 bytes | **116 bytes** |
| Compression | 1× | 1× | **2×** |
| Quant delta | — | — | **0.00%** |

> **Key insight:** PC sim (95.3%) uses the trained MLP with feature extraction.
> PC feed (57.9%) uses the threshold classifier (same as the dry-run simulated ESP32).
> Once `train_simple_model.py` → `export_rust_weights.py` → reflash, ESP32 accuracy rises.

---

*Educational embedded medical prototype — not for clinical use.*
```
RT-QTinyECG-ESP32-Rust/
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/                        ← generated data (git-ignored .csv/.pkl)
│   ├── sample_ecg.csv           ← synthetic ECG + ground truth labels
│   ├── simulated_realtime_log.csv
│   ├── quantized_weights.npz
│   ├── model.pkl / scaler.pkl
│   ├── pc_predictions.csv       ← from Phase 3
│   ├── esp32_predictions.csv    ← from Phase 3
│   └── comparison_report.csv   ← from Phase 3
│
├── python/
│   ├── generate_dummy_ecg.py    ← Phase 1: data generation
│   ├── preprocessing.py
│   ├── train_simple_model.py    ← Phase 1: train float32 model
│   ├── realtime_ecg_simulator.py← Phase 1: PC simulation
│   ├── quantize_weights.py      ← convert float32 → int8
│   ├── export_rust_weights.py   ← write model_weights.rs
│   ├── metrics.py               ← Phase 1: evaluate simulation
│   ├── uart_feed_evaluator.py   ← Phase 3: feed ESP32 via UART [NEW]
│   ├── compare_models.py        ← Phase 3: cross-model comparison [NEW]
│   ├── optimization_report.py   ← Phase 4: suggest optimizations [NEW]
│   └── fine_tune_model.py       ← Phase 4: retrain on real data [NEW]
│
├── firmware/esp32-rust/
│   ├── Cargo.toml               ← esp-hal 1.1, xtensa-lx-rt 0.22
│   ├── rust-toolchain.toml      ← Xtensa Rust 1.95
│   ├── .cargo/config.toml       ← build-std, xtensa target
│   └── src/
│       ├── main.rs              ← entry point (UART_FEED_MODE flag)
│       ├── inference.rs         ← int8 MLP + threshold classifier
│       ├── model_weights.rs     ← quantized weights (auto-generated)
│       ├── ring_buffer.rs       ← circular buffer
│       ├── filter.rs            ← moving average filter
│       ├── logger.rs            ← UART CSV output
│       ├── uart_feed.rs         ← UART input mode [NEW]
│       ├── adc_reader.rs        ← ADC mode (reference)
│       └── alert.rs             ← GPIO alert (reference)
│
├── gnuplot/
│   ├── plot_ecg_signal.gp
│   ├── plot_alert_latency.gp
│   └── plot_inference_time.gp
│
├── images/                      ← generated PNG charts
│
└── docs/                        ← you are here
    ├── INDEX.md                 ← this file
    ├── block_diagram.md
    ├── wiring_esp32_ad8232.md
    ├── flowchart.md
    ├── algorithm.md
    ├── real_time_design.md
    ├── evaluation_metrics.md
    ├── uart_feed_evaluation.md  ← UART pipeline steps
    └── optimization_guide.md   ← TinyML optimization reference
```

---

## Current Status

| Component | Status |
|-----------|--------|
| Python simulation pipeline | ✅ Working |
| ESP32 firmware build | ✅ Builds clean (`67,600 bytes`) |
| UART Feed Evaluator script | 🔲 Not yet implemented |
| Cross-model comparison script | 🔲 Not yet implemented |
| UART firmware mode (`UART_FEED_MODE`) | 🔲 Not yet implemented |
| Optimization report script | 🔲 Not yet implemented |
| Fine-tune script | 🔲 Not yet implemented |

---

## Current Metrics (Simulation Baseline)

From `python/metrics.py` run on `data/simulated_realtime_log.csv`:

| Metric | Value |
|--------|-------|
| Sample rate | 250 Hz |
| Avg inference time (PC) | ~12 µs |
| PC float32 accuracy | 62% (threshold classifier, room to improve) |
| ESP32 int8 model size | 116 bytes |
| Total static RAM | 724 bytes |
| Alert latency | 1.5 ms |

> **Note:** The 62% accuracy is from the threshold classifier, not the trained MLP.
> Run `train_simple_model.py` → `export_rust_weights.py` → rebuild to use the
> trained MLP (~85–90% expected).

---

*Educational embedded medical prototype — not for clinical use.*
