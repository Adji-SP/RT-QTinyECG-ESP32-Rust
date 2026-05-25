# Algorithm Description

This document describes the signal-processing and classification algorithm used by the current RT-QTinyECG-ESP32-Rust code.

## Processing Pipeline

```text
ADC or UART sample
    -> moving average filter
    -> 128-sample ring buffer
    -> feature extraction
    -> firmware-style feature normalization
    -> threshold classifier or int8 MLP
    -> alert output and logging
```

The production firmware path uses a quantized MLP by default. The threshold classifier remains in `inference.rs` as a transparent fallback for debugging.

## Sampling

Nominal sampling rate:

```text
250 Hz = 4 ms per sample
```

Input sources:

| Mode | Source |
|---|---|
| ADC mode | ESP32-S3 ADC input on GPIO4 |
| UART-feed mode | PC sends ASCII ADC values over UART0 |
| Dry-run mode | Python simulates the ESP32/int8 path from `quantized_weights.npz` |

ADC values are treated as 12-bit samples:

```text
0..4095
```

The synthetic dataset is centered near `2048`, which represents the midpoint of a 3.3 V ADC range.

## Moving Average Filter

The firmware applies a causal moving average filter with an 8-sample window:

```text
filtered[n] = average(x[n], x[n-1], ..., x[n-7])
```

The Rust implementation keeps a running sum, so per-sample update cost is O(1):

```rust
running_sum = running_sum + new_sample - oldest_sample;
average = running_sum / count;
```

This smooths high-frequency noise and keeps implementation cost small enough for a microcontroller.

## Ring Buffer

The latest 128 filtered samples are stored in a fixed-size ring buffer:

```text
128 samples / 250 Hz = 512 ms window
```

After the buffer fills, inference can run every new sample. Training now uses the same cadence: one feature vector per sample after the first 128 samples.

Memory cost:

```text
128 * sizeof(i32) = 512 bytes
```

## Feature Extraction

For each full window, firmware computes:

| Feature | Formula |
|---|---|
| Mean | `sum(x) / N` |
| Maximum | `max(x)` |
| Minimum | `min(x)` |
| Peak-to-peak | `maximum - minimum` |
| Energy | `sum(x*x) / N` |

The MLP input vector is:

```text
[mean, maximum, minimum, peak_to_peak, energy / 4096]
```

Then it is normalized per window to an int8-like range:

```text
feat_max = max(abs(each feature), 1)
feat_q[i] = clamp(feature[i] * 127 / feat_max, -128, 127)
```

The Python training code uses the same transformation via `extract_firmware_mlp_input_array()` in `python/preprocessing.py`.

## Threshold Classifier

The threshold classifier is simple and interpretable:

```text
if peak_to_peak > 600 -> abnormal
if mean > 2350        -> abnormal
if mean < 1750        -> abnormal
else                  -> normal
```

These constants are in `firmware/esp32-rust/src/inference.rs`:

```rust
const THRESH_P2P: i32 = 600;
const THRESH_MEAN_HIGH: i32 = 2350;
const THRESH_MEAN_LOW: i32 = 1750;
```

This mode is useful for debugging, but the current default inference mode is the quantized MLP.

## Quantized MLP

Architecture:

```text
5 inputs -> 8 hidden ReLU units -> 1 output
```

Weight layout in Rust:

```text
W1: [8 x 5] row-major, indexed as W1[hidden * 5 + feature]
B1: [8]
W2: [1 x 8], stored as [8] because there is one output
B2: [1]
```

sklearn stores first-layer weights as `[features, hidden]`. `export_rust_weights.py` transposes this layout before writing `model_weights.rs`.

Quantization:

```text
scale = max(abs(W)) / 127
W_q = round(W / scale), clamped to int8
B_q = round(B / scale), stored as i32
```

Firmware inference:

```text
hidden[j] = relu(sum_i(W1[j,i] * feat_q[i]) + B1[j])
hidden_q[j] = hidden[j] * 127 / max(hidden)
output = sum_j(W2[j] * hidden_q[j]) + B2[0]
prediction = 1 if output > 0 else 0
```

## Training and Export

The current retraining flow is:

```bat
py python\generate_dummy_ecg.py
py python\train_simple_model.py
py python\quantize_weights.py
py python\export_rust_weights.py
```

Important training details:

- Training uses firmware-compatible feature normalization.
- The current `scaler.pkl` contains `None`; it is retained for compatibility.
- The generated Rust weights are written to `firmware/esp32-rust/src/model_weights.rs`.

## Alert Logic

Firmware maps prediction to outputs:

| Prediction | Meaning | LED | Buzzer |
|---:|---|---|---|
| `0` | Normal | Low | Low |
| `1` | Abnormal | High | High |

GPIO mapping for ESP32-S3:

| Output | GPIO |
|---|---:|
| LED | GPIO2 |
| Buzzer | GPIO21 |

## Current Performance

Latest dry-run evaluation:

| Metric | Value |
|---|---:|
| PC float32 accuracy | 96.5% |
| ESP32/int8 dry-run accuracy | 95.4% |
| Quantization delta | 1.05% |
| Agreement | 98.9% |

These metrics come from synthetic ECG-like data and are not clinical validation.

