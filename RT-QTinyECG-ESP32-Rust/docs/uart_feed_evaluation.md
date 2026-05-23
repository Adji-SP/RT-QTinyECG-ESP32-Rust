# UART Feed Evaluation — Real-Time Cross-Model Validation

## Overview

This document describes the **UART Feed Evaluation** pipeline — an experimental
framework that replaces the real AD8232 ECG sensor with PC-generated simulated
data, feeds it to the ESP32 via UART, captures the ESP32's embedded int8
inference results, and compares them against a reference float32 model running
on the laptop. The result is a controlled, reproducible accuracy benchmark for
quantized TinyML on embedded hardware.

> **No real sensor required.** No Proteus. No analog noise variables.
> The same synthetic ECG dataset is evaluated by both models simultaneously,
> giving a ground-truth comparison of quantization degradation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      PC  (Python)                               │
│                                                                 │
│  sample_ecg.csv                                                 │
│  (2500 samples + GT labels)                                     │
│         │                                                       │
│         ├──────────────────────────────────────────────────────│
│         │   Step A: PC float32 sklearn model                    │
│         │   → pc_predictions.csv                               │
│         │                                                       │
│         │   Step B: Send samples via UART TX ──────────────┐   │
│         │                                                   │   │
│         │   Step C: Receive ESP32 predictions via UART RX ◄─┤   │
│         │   → esp32_predictions.csv                         │   │
│         │                                                   │   │
│         └── Step D: Compare A vs C vs GT labels            │   │
│                     → comparison_report.csv                 │   │
│                     → accuracy delta, F1, confusion matrix  │   │
└─────────────────────────────────────────────────────────────│───┘
                                                              │ USB/UART
                              ┌───────────────────────────────▼──┐
                              │         ESP32 (Rust firmware)     │
                              │                                   │
                              │  UART RX ──→ int8 inference       │
                              │             (model_weights.rs)    │
                              │             ──→ UART TX           │
                              │             prediction: 0 or 1    │
                              └───────────────────────────────────┘
```

---

## Why This Approach

| Property | Real Sensor | UART Feed Mode |
|----------|-------------|----------------|
| Ground truth available | ✗ (unlabeled) | ✅ (synthetic, labeled) |
| Reproducible runs | ✗ | ✅ identical every run |
| Controlled experiment | ✗ (noise varies) | ✅ clean synthetic data |
| Quantization measurement | Approximate | ✅ exact (same input) |
| Hardware required | AD8232 + ESP32 | ESP32 only |

---

## Implementation Steps

### Step 0 — Prerequisites

```powershell
# Python packages
pip install pyserial scikit-learn numpy pandas

# Rust firmware already built (see README_FIRMWARE.md)
# ESP32 connected via USB
# Find your COM port:
python -c "import serial.tools.list_ports; [print(p) for p in serial.tools.list_ports.comports()]"
```

---

### Step 1 — Generate Labeled Simulated ECG Dataset

```powershell
$env:PYTHONIOENCODING='utf-8'
py python/generate_dummy_ecg.py
```

**Output:** `data/sample_ecg.csv`

```
# Format: time_ms, adc_value, ground_truth_label
0,2048,0
4,2051,0
...
24000,3800,1    ← abnormal segment begins at 6s
```

---

### Step 2 — Run PC Reference Model (float32)

```powershell
py python/train_simple_model.py
py python/realtime_ecg_simulator.py
```

This runs the full float32 sklearn pipeline on the simulated data.

**Output:** `data/simulated_realtime_log.csv`
Columns: `time_ms, adc_raw, filtered, inference_us, prediction, alert, alert_latency_ms`

Save the PC predictions separately:

```powershell
py python/extract_pc_predictions.py   # (new script — see Step 5)
```

**Output:** `data/pc_predictions.csv`

---

### Step 3 — Flash UART-Feed Firmware to ESP32

The firmware needs a compile-time flag to switch from **ADC mode** to **UART feed mode**.
In UART feed mode, the ESP32:
- Reads ADC integer values from UART RX (one per line, e.g. `"2048\n"`)
- Runs inference
- Sends back the prediction as `"0\n"` or `"1\n"`

#### 3a. Enable UART feed mode in firmware

In `firmware/esp32-rust/src/main.rs`, set:

```rust
// Set to true to enable UART feed mode (no ADC needed)
const UART_FEED_MODE: bool = true;
```

#### 3b. Rebuild and flash

```powershell
cd firmware/esp32-rust
. $HOME\export-esp.ps1
cargo build --release
espflash flash target\xtensa-esp32-none-elf\release\ecg-esp32 --monitor
```

---

### Step 4 — Run UART Feed Evaluator

```powershell
py python/uart_feed_evaluator.py --port COM3 --baud 115200
```

**What this script does:**
1. Opens serial port to ESP32
2. Reads `data/sample_ecg.csv` line by line
3. Sends each ADC value to ESP32: `"2048\n"`
4. Waits for ESP32 to respond with prediction: `"0\n"` or `"1\n"`
5. Records both into `data/esp32_predictions.csv`

**Output:** `data/esp32_predictions.csv`

```
# Format: time_ms, adc_value, gt_label, esp32_prediction, round_trip_ms
0,2048,0,0,8.2
4,2051,0,0,8.1
...
```

---

### Step 5 — Cross-Model Validation

```powershell
py python/compare_models.py
```

**What this script does:**
1. Loads `data/simulated_realtime_log.csv` (PC float32 predictions)
2. Loads `data/esp32_predictions.csv` (ESP32 int8 predictions)
3. Loads `data/sample_ecg.csv` (ground truth labels)
4. Computes:
   - Accuracy of PC model vs ground truth
   - Accuracy of ESP32 model vs ground truth
   - **Quantization accuracy delta** (PC − ESP32)
   - **Model disagreement rate** (when PC and ESP32 predict differently)
   - UART round-trip latency statistics

**Output:** `data/comparison_report.csv` + terminal summary

```
═══════════════════════════════════════════════════
  Cross-Model Validation Report
═══════════════════════════════════════════════════

  PC  float32 accuracy  : 0.94
  ESP32 int8  accuracy  : 0.91
  Quantization delta    : -0.03  (3% accuracy loss from float→int8)

  Model agreement rate  : 96.4%  (3.6% samples where they disagree)
  Disagreements @ normal: 12 samples
  Disagreements @ abnorm: 78 samples  ← focus for optimization

  Mean UART round-trip  : 8.3 ms
  Max  UART round-trip  : 14.7 ms
═══════════════════════════════════════════════════
```

---

### Step 6 — Optimization Analysis

After the comparison, identify optimization options:

```powershell
py python/optimization_report.py
```

See [optimization_guide.md](optimization_guide.md) for the full optimization
strategy document.

Key outputs:
- **Where do models disagree?** → samples near the threshold boundary
- **Is the threshold too aggressive?** → adjust `THRESHOLD` in `inference.rs`
- **Should we retrain on boundary samples?** → fine-tuning loop

---

### Step 7 — Fine-Tuning Loop (Optional)

If accuracy delta is too high (> 5%), retrain the PC model with real-feeling data
and re-quantize:

```powershell
# 1. Augment training data with boundary samples
py python/fine_tune_model.py --epochs 50

# 2. Re-quantize with refined weights
py python/quantize_weights.py

# 3. Export new weights to Rust
py python/export_rust_weights.py
# → overwrites firmware/esp32-rust/src/model_weights.rs

# 4. Rebuild firmware
cd firmware/esp32-rust
cargo build --release
espflash flash target\xtensa-esp32-none-elf\release\ecg-esp32

# 5. Re-run evaluation (Step 4 + 5)
# → compare new accuracy delta
```

---

## Data Flow Summary

```
generate_dummy_ecg.py
        │
        ├── sample_ecg.csv (2500 samples + GT labels)
        │         │
        │         ├── [PC path]  realtime_ecg_simulator.py
        │         │              → simulated_realtime_log.csv (float32 preds)
        │         │
        │         └── [ESP32 path]  uart_feed_evaluator.py → serial → ESP32
        │                           → esp32_predictions.csv (int8 preds)
        │
        └── compare_models.py
                  │
                  ├── comparison_report.csv
                  ├── Terminal summary (accuracy delta, disagreement rate)
                  └── optimization_report.py → next optimization target
```

---

## Files Involved

### Existing (already built)
| File | Role |
|------|------|
| `python/generate_dummy_ecg.py` | Generate labeled ECG dataset |
| `python/train_simple_model.py` | Train float32 sklearn model |
| `python/realtime_ecg_simulator.py` | PC float32 inference pipeline |
| `python/quantize_weights.py` | Quantize float32 → int8 |
| `python/export_rust_weights.py` | Export weights to Rust |
| `python/metrics.py` | Evaluation metrics |
| `firmware/esp32-rust/src/main.rs` | ESP32 firmware entry point |
| `firmware/esp32-rust/src/inference.rs` | Embedded int8 classifier |

### New (to be implemented)
| File | Role |
|------|------|
| `python/uart_feed_evaluator.py` | Feed data to ESP32, capture predictions |
| `python/compare_models.py` | Cross-model accuracy comparison |
| `python/optimization_report.py` | Identify optimization targets |
| `python/fine_tune_model.py` | Retrain on boundary-case samples |
| `firmware/esp32-rust/src/uart_feed.rs` | UART input mode for firmware |

---

## Expected Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| PC float32 accuracy | Baseline sklearn model | > 90% |
| ESP32 int8 accuracy | Embedded quantized model | > 85% |
| Quantization delta | Accuracy lost from float→int8 | < 5% |
| Model agreement rate | Samples where both agree | > 95% |
| UART round-trip latency | PC→ESP32→PC per sample | < 20 ms |
| ESP32 inference time | Time for one inference on-chip | < 100 µs |

---

## Troubleshooting

**ESP32 not responding:**
```powershell
# Check port
python -c "import serial.tools.list_ports; [print(p.device) for p in serial.tools.list_ports.comports()]"
# Reset ESP32 manually (press EN button) then re-run evaluator
```

**UART timeouts (data loss):**
- Reduce sample rate from 250 Hz to 125 Hz (`SAMPLE_RATE_HZ = 125`)
- Or reduce UART baud to avoid Windows serial buffer issues

**High quantization delta (> 10%):**
- Run `optimization_report.py` to identify problematic samples
- Fine-tune model with Step 7 above
- Check `THRESHOLD` value in `inference.rs`

---

*This pipeline is for educational embedded systems research only.
Not a clinical medical device.*
