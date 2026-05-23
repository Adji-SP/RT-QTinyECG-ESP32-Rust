# Algorithm Description

This document explains the signal processing and classification algorithms
used in the RT-QTinyECG-ESP32-Rust system.

---

## Overview

The system implements a lightweight real-time ECG signal processing pipeline:

```
ADC Sample → Moving Average → Ring Buffer → Feature Extraction → Classifier → Alert
```

Each component is designed for:
- **No heap allocation** (all buffers on stack or static memory)
- **Deterministic execution time** (no dynamic dispatch, no malloc)
- **Integer arithmetic** where possible (avoids FPU overhead)
- **Educational clarity** over production optimization

---

## 1. ADC Sampling

**What it does:**
Reads the ECG analog voltage from GPIO34 (ADC1_CH6) at a fixed rate.

**How it works:**
- ESP32 ADC1 performs a successive approximation conversion
- 12-bit resolution: output range 0–4095
- 0 → 0V, 4095 → 3.3V (approximately)
- Attenuation: 11 dB → full 0–3.3V input range

**Fixed-rate sampling:**
The main loop calls `delay_micros(4000)` to achieve approximately 250 Hz.
This is a busy-wait approach. For better accuracy, use a hardware timer ISR.

**Why 250 Hz?**
ECG signals contain energy up to ~150 Hz. Nyquist theorem requires
sampling at ≥ 300 Hz. At 250 Hz, we capture most ECG content.
For educational demos, 250 Hz is sufficient.

---

## 2. Moving Average Filter

**What it does:**
Smooths the raw ADC signal to reduce high-frequency noise.

**Algorithm:**
```
filtered[i] = (adc[i] + adc[i-1] + ... + adc[i-N+1]) / N
```

Where N = 8 (filter window size).

**Implementation:**
The `MovingAverageState<N>` struct maintains a running sum for O(1) update:
```rust
running_sum = running_sum + new_sample - oldest_sample
average     = running_sum / count
```

**Effect on frequency:**
At 250 Hz with window=8, the filter cutoff is approximately:
- 3 dB cutoff ≈ 0.443 × fs / N ≈ 0.443 × 250 / 8 ≈ 13.8 Hz
- This attenuates noise above ~14 Hz while preserving the ECG shape

**Trade-off:**
- Larger window → smoother but more lag
- Smaller window → faster but noisier
- Window=8 is a good compromise for 250 Hz ECG

---

## 3. Ring Buffer (Sliding Window)

**What it does:**
Stores the last 128 filtered samples for inference.

**Why 128 samples?**
128 samples at 250 Hz = 512 ms window.
This captures approximately 1 complete heartbeat at 72 BPM (833 ms/beat).
Slightly less than one beat, but sufficient for amplitude and baseline analysis.

For true beat-to-beat analysis, increase to 256 samples (1024 ms).

**How it works:**
Circular buffer with a head pointer. When full, oldest sample is overwritten:
```
[s0, s1, s2, ..., s127]  ← fixed 128-element array
     ↑head               ← next write position (wraps around)
```

**Memory usage:**
128 × sizeof(i32) = 128 × 4 = **512 bytes** (negligible on ESP32).

---

## 4. Feature Extraction

**What it does:**
Converts 128 raw samples into 5 scalar features for the classifier.

**Features computed:**

| Feature | Formula | Why it matters |
|---|---|---|
| Mean | Σx / N | Baseline level; elevated mean = ST elevation |
| Maximum | max(x) | Peak amplitude |
| Minimum | min(x) | Trough amplitude |
| Peak-to-peak | max - min | Beat amplitude; high = erratic |
| Energy | Σ(x²) / N | Signal power; changes with arrhythmia |

All computed using integer arithmetic in O(N) time.

---

## 5. Threshold Classifier

**What it does:**
Applies three simple rules to classify the ECG window as Normal or Abnormal.

**Rules:**

```
IF peak_to_peak > 600:   → Abnormal (high amplitude = irregular beats)
IF mean > 2350:          → Abnormal (elevated baseline = ST elevation simulation)
IF mean < 1750:          → Abnormal (depressed baseline = signal loss)
ELSE:                    → Normal
```

**Why these thresholds?**
- ADC midpoint = 2048 (corresponds to 1.65V, AD8232 resting output)
- Normal ECG at 3.3V supply has amplitude of ~0.3–0.5V → ~150–250 ADC units peak-to-peak
- We set the threshold at 600 to allow some margin and detect significant changes
- Baseline thresholds (±300 from midpoint) detect significant baseline shift

**Tuning:**
Adjust `THRESH_P2P`, `THRESH_MEAN_HIGH`, `THRESH_MEAN_LOW` in `src/inference.rs`.
After adjusting, observe output on serial monitor and tune based on your signal.

---

## 6. Quantized Tiny MLP Classifier (Optional)

**Architecture:**
```
Input:  5 features (mean, max, min, peak-to-peak, energy)
Layer 1: 5 → 8 neurons, ReLU, int8 weights
Layer 2: 8 → 1 neuron, linear, int8 weights
Output: 0 or 1 (threshold at 0)
```

**Quantization:**
Float32 weights from training are quantized to int8:
```
scale     = max(|W|) / 127.0
W_int8[i] = clip(round(W_f32[i] / scale), -128, 127)
```

**Inference in integer arithmetic:**
```rust
// Layer 1 (no float):
hidden[j] = relu( Σ_i( W1[j,i]_i8 × feat_q[i]_i32 ) + B1[j]_i32 )

// Layer 2:
output = Σ_j( W2[j]_i8 × hidden_q[j]_i32 ) + B2_i32

// Decision:
if output > 0 → Abnormal (1)
else          → Normal   (0)
```

**Memory:**
- W1: 40 bytes (i8)
- B1: 32 bytes (i32)
- W2: 8 bytes  (i8)
- B2: 4 bytes  (i32)
- **Total: 84 bytes** → fits easily in ESP32 flash + DRAM

---

## 7. Alert Logic

**State machine:**
```
If prediction == 1 (Abnormal):
    If alert was OFF: record alert_start_time
    Turn LED and Buzzer ON
    Compute latency = current_time - alert_start_time

If prediction == 0 (Normal):
    Turn LED and Buzzer OFF
    Reset alert_start_time
```

**Alert latency:**
The time from "inference output = 1" to "GPIO HIGH" is approximately
the GPIO write time (~2–5 µs on ESP32 at 240 MHz).
In the current busy-wait design, no other code runs between inference and GPIO write.

---

## 8. UART CSV Logging

**Purpose:**
Every sample is logged over UART for post-processing and visualization.

**Format:**
```
time_ms,adc_value,filtered_value,inference_us,prediction,alert,alert_latency_ms
```

**Bandwidth note:**
At 115200 baud and ~60 chars/line, each line takes ~5.2 ms to transmit.
This exceeds the 4 ms sampling interval.

**Solutions implemented:**
1. Use `log_csv_throttled()` to log every Nth sample
2. Increase UART baud to 921600 for <1 ms/line
3. Use compact binary format (not implemented in this demo)

---

## Algorithm Summary

| Component | Complexity | Latency (ESP32, 240 MHz) |
|---|---|---|
| ADC read | O(1) | ~20–100 µs |
| Moving average | O(1) | ~1 µs |
| Ring buffer push | O(1) | ~0.1 µs |
| Feature extraction | O(N) | ~5–10 µs |
| Threshold classify | O(1) | ~1 µs |
| MLP classify (5→8→1) | O(N×M) | ~20–50 µs |
| GPIO write (alert) | O(1) | ~2 µs |
| UART log (115200) | O(L) | ~5 ms |
