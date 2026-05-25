# Evaluation Metrics Guide

This document explains the metrics used to evaluate the PC float32 model, the ESP32/int8 model, and the real-time behavior of the prototype.

## Main Evaluation Commands

Dry-run evaluation:

```bat
cd /d D:\PropertiesProject-D\Kuliah\Pemkon\Product
run_uart_eval.bat --dry-run
```

Hardware evaluation:

```bat
run_uart_eval.bat COM16
```

Regenerate comparison only:

```bat
py python\compare_models.py
py python\optimization_report.py
```

## Current Dry-Run Results

| Metric | PC float32 | ESP32/int8 dry-run |
|---|---:|---:|
| Accuracy | 96.5% | 95.4% |
| Precision | 1.000 | 1.000 |
| Recall | 0.917 | 0.892 |
| F1-score | 0.957 | 0.943 |
| Quantization delta | - | 1.05% |
| Agreement | - | 98.9% |

These values are from synthetic ECG-like data and are not clinical validation.

## Classification Metrics

Confusion matrix:

```text
                    Predicted normal   Predicted abnormal
Actual normal              TN                  FP
Actual abnormal            FN                  TP
```

| Metric | Formula | Meaning |
|---|---|---|
| Accuracy | `(TP + TN) / total` | Overall correctness |
| Precision | `TP / (TP + FP)` | How often abnormal predictions are correct |
| Recall | `TP / (TP + FN)` | How many abnormal samples are detected |
| F1-score | `2 * precision * recall / (precision + recall)` | Balance between precision and recall |
| Agreement | `PC prediction == ESP32 prediction` | Model alignment |
| Quantization delta | `PC accuracy - ESP32 accuracy` | Accuracy loss after int8 quantization |

Targets for this educational project:

| Metric | Good | Acceptable |
|---|---:|---:|
| PC accuracy | > 90% | > 85% |
| ESP32/int8 accuracy | > 90% | > 85% |
| Quantization delta | < 3% | < 5% |
| Agreement | > 95% | > 90% |

## Real-Time Metrics

### Sampling Interval

Expected interval:

```text
1000 ms / 250 Hz = 4 ms
```

Metrics:

| Metric | Target |
|---|---:|
| Mean interval | 4.0 ms |
| Standard deviation | As low as practical |
| Coefficient of variation | < 5% |

UART logging at 115200 baud can block longer than the 4 ms sampling interval if a full CSV line is sent every sample.

### Inference Time

Expected embedded compute time:

| Classifier | Expected time |
|---|---:|
| Threshold | 5-20 us |
| Int8 MLP 5 -> 8 -> 1 | 20-60 us |

The firmware currently uses a placeholder `25 us` timing value in the main loop. For real measurements, add a CPU cycle counter around `inference::infer()`.

### Alert Latency

Important components:

| Component | Approximate latency |
|---|---:|
| Initial 128-sample window fill | 512 ms |
| Feature extraction and inference | Tens of microseconds |
| GPIO toggle | Microseconds |
| UART logging | Depends on baud rate |

The dominant delay is the window length, not the MLP computation.

## Model Size

Current 5 -> 8 -> 1 quantized model:

| Item | Size |
|---|---:|
| W1 int8 `[8 x 5]` | 40 bytes |
| B1 int32 `[8]` | 32 bytes |
| W2 int8 `[1 x 8]` | 8 bytes |
| B2 int32 `[1]` | 4 bytes |
| Scale metadata | 16 bytes |
| Total with metadata | 100 bytes |

`optimization_report.py` reports 116 bytes because it sums the `.npz` arrays, including NumPy scalar storage overhead for metadata.

## Reports and Artifacts

| File | Meaning |
|---|---|
| `data/esp32_predictions.csv` | PC feed and ESP32/dry-run predictions |
| `data/comparison_report.csv` | Per-sample model comparison |
| `data/optimization_targets.json` | Metrics and recommendations |
| `data/plots/*.csv` | GNUPlot-ready data |
| `images/evaluation/*.png` | Visual comparison charts |

## Interpreting Failures

| Symptom | Likely cause |
|---|---|
| PC accuracy good, ESP32 accuracy bad | Feature, scaler, or weight-layout mismatch |
| Quantization delta > 5% | Quantization or integer inference mismatch |
| Many `PC=normal, ESP32=abnormal` | ESP32 over-detecting; inspect thresholds/layout |
| Many `PC=abnormal, ESP32=normal` | ESP32 under-detecting; model may be too conservative |
| UART timeouts | Wrong port, monitor open, baud mismatch, firmware not in UART-feed mode |

