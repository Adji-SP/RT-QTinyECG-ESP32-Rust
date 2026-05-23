# Evaluation Metrics Guide

This document explains how to measure, interpret, and improve the key metrics
of the RT-QTinyECG-ESP32-Rust system.

Run `python python/metrics.py` to compute all metrics automatically from the simulation log.

---

## 1. Sampling Interval Stability

**What it measures:**
How consistently the ESP32 samples the ECG signal at 250 Hz (4 ms intervals).
Jitter in sampling intervals causes spectral aliasing and timing errors.

**How to measure:**
- Log `time_ms` for each sample
- Compute intervals: `Δt[i] = time_ms[i] - time_ms[i-1]`
- Statistics: mean, standard deviation, coefficient of variation (CV)

**Metrics:**
| Metric | Formula | Good Target |
|---|---|---|
| Mean interval | `mean(Δt)` | 4.0 ms |
| Std deviation | `std(Δt)` | < 0.2 ms |
| CV (jitter %) | `std(Δt) / mean(Δt)` | < 5% |

**On real ESP32:**
Busy-wait delay (`delay_micros()`) has jitter from:
- UART TX blocking time (~5 ms at 115200 baud) → **biggest issue**
- ADC conversion time (~20–100 µs)
- Inference time (~5–50 µs)

**To improve:**
1. Use a hardware timer ISR for sampling (not busy-wait)
2. Reduce UART logging frequency or increase baud rate
3. Log `time_ms` from a hardware counter, not software increment

---

## 2. Inference Time

**What it measures:**
How long the classifier takes to process one ECG window.
Must be much less than the 4 ms sampling interval.

**How to measure on ESP32:**
Use the Xtensa CCOUNT cycle counter register:
```rust
let start = read_ccount();  // Xtensa: core::arch::asm!("rsr.ccount {0}", ...)
inference::infer(window);
let end = read_ccount();
let cycles = end.wrapping_sub(start);
let us = cycles / 240;  // at 240 MHz
```

**Metrics:**
| Metric | Formula | Good Target |
|---|---|---|
| Mean inference time | `mean(inference_us)` | < 500 µs |
| Max inference time | `max(inference_us)` | < 3000 µs |
| P95 inference time | 95th percentile | < 1000 µs |
| Budget utilization | `max_time / 4000` | < 25% |

**Expected values:**
- Threshold classifier: 5–20 µs
- MLP (5→8→1, int8): 20–60 µs

**On PC simulator:**
Python's `time.perf_counter()` measures wall-clock time including OS scheduling.
PC values are much larger than embedded values. Use only for relative comparison.

---

## 3. Alert Latency

**What it measures:**
Time from "inference detects abnormal" to "alert GPIO goes HIGH".

**Why it matters:**
In a real safety alert system, long latency could delay response to a cardiac event.
For this educational prototype, latency should be well below 10 ms.

**Components of latency:**
1. **Window fill delay**: Must wait for 128 samples before first inference
   - At 250 Hz: 128 / 250 = 512 ms (fixed, unavoidable)
2. **Inference time**: 5–50 µs (negligible)
3. **GPIO write time**: ~2–5 µs (negligible)
4. **UART blocking**: Up to 5 ms (avoidable with higher baud rate)

**Note:**
The 512 ms window-fill latency is the dominant delay.
It can be reduced by using a smaller window (e.g., 64 samples = 256 ms).
But smaller windows reduce feature quality.

**Metrics:**
| Metric | Formula | Target |
|---|---|---|
| GPIO toggle latency | `GPIO_high_time - inference_complete_time` | < 5 µs |
| Total system latency | window fill + inference + GPIO | < 600 ms |
| Alert duration jitter | std of alert ON times | < 1 ms |

---

## 4. Classification Metrics

**What they measure:**
How accurately the classifier identifies Normal vs. Abnormal ECG.

**On simulated data:**
```
python python/metrics.py
```

**Metrics explained:**

| Metric | Formula | Best Value |
|---|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | 1.0 |
| Precision | TP / (TP + FP) | 1.0 |
| Recall (Sensitivity) | TP / (TP + FN) | 1.0 |
| Specificity | TN / (TN + FP) | 1.0 |
| F1-Score | 2 × P × R / (P + R) | 1.0 |

**For medical prototype tradeoffs:**
- **High Recall** (catch all abnormals) → minimize false negatives (FN)
  - Accept more false alarms (lower precision)
- **High Precision** (only alarm when truly abnormal) → minimize false positives (FP)
  - May miss some abnormals (lower recall)

For this educational demo, we prioritize **recall** (don't miss abnormals).

**Confusion matrix:**
```
                 Predicted Normal | Predicted Abnormal
Actual Normal        TN           |       FP
Actual Abnormal      FN           |       TP
```

---

## 5. Model Size

**What it measures:**
Flash memory required to store the quantized model weights.

**Components:**

| Component | Bytes | Notes |
|---|---|---|
| W1 weights (i8) | 40 | 8 × 5 = 40 values |
| B1 biases (i32) | 32 | 8 × 4 bytes |
| W2 weights (i8) | 8 | 1 × 8 = 8 values |
| B2 bias (i32) | 4 | 1 × 4 bytes |
| Scale factors | 16 | 4 × f32 = optional |
| **Total** | **~100 bytes** | Negligible |

Compared to:
- TensorFlow Lite Micro: minimum ~20 KB just for runtime
- Float32 weights: ~4× larger than int8

**ESP32 Flash: 4 MB** → model is 100 bytes / 4,194,304 bytes = **0.0024%** of flash.

---

## 6. Memory Usage (RAM)

**Static RAM usage estimate:**

| Variable | Size | Notes |
|---|---|---|
| Ring buffer `[i32; 128]` | 512 bytes | 128 × 4 |
| Filter state `[i32; 8]` | 32 bytes | Moving avg window |
| Log buffer (string) | ~64 bytes | UART output buffer |
| Model weights (static) | ~100 bytes | In Flash, not RAM |
| Stack frame | ~2–4 KB | Per call frame |
| **Total (est.)** | **~5 KB** | |

**ESP32 DRAM: 520 KB** → usage is **~1%** of available RAM.

---

## 7. Current Consumption Estimation

> Note: Not measured in this software prototype. Estimated from ESP32 datasheet.

| Mode | Current | Notes |
|---|---|---|
| Normal operation (WiFi OFF) | ~80–100 mA | Continuous ADC + UART |
| Light sleep (partial) | ~800 µA | Not used in this demo |
| Deep sleep | ~10 µA | Not compatible with continuous sampling |

**Battery life estimate (1000 mAh LiPo @ 3.7V):**
- At 90 mA: 1000 / 90 ≈ **11 hours** continuous operation
- At 250 Hz sampling with WiFi OFF: approximately 8–12 hours

---

## 8. Summary Table (Expected Results on PC Simulator)

| Metric | Expected Value | Unit |
|---|---|---|
| Sampling interval mean | 4.0 | ms |
| Sampling interval std | < 0.1 | ms |
| Inference time (threshold) | 1–10 | µs (PC) |
| Inference time (MLP) | 5–50 | µs (PC) |
| Alert latency (GPIO, est.) | 1.5 | ms |
| Accuracy | > 90% | % |
| F1-Score | > 0.85 | – |
| Model size | ~100 | bytes |
| Total static RAM | ~5 | KB |
| Sampling budget used | < 1% | % |
