# Optimization Guide — TinyML on ESP32

## Overview

After running the UART Feed Evaluation pipeline, this guide explains **how to
analyze and act on the results** to improve the embedded model's accuracy,
efficiency, or resource footprint on the ESP32.

---

## Optimization Decision Tree

```
Run compare_models.py
        │
        ├── Quantization delta > 5%?
        │       └── YES → Fine-tune model (see Section 3)
        │               → Adjust quantization scale (Section 4)
        │
        ├── Accuracy < 85%?
        │       └── YES → Adjust threshold (Section 1)
        │               → Improve feature window (Section 5)
        │
        ├── Disagreement mostly on "abnormal" samples?
        │       └── YES → Threshold too high (Section 1)
        │               → Model misses abnormal patterns (Section 3)
        │
        ├── Inference time > 500 µs?
        │       └── YES → Simplify architecture (Section 2)
        │               → Reduce window size (Section 5)
        │
        └── RAM > 50 KB?
                └── YES → Shrink buffer size (Section 5)
                        → Use int4 instead of int8 (Section 4)
```

---

## Section 1 — Threshold Adjustment (Quickest Fix)

The current classifier uses a simple threshold or MLP with a decision boundary.

**File:** `firmware/esp32-rust/src/inference.rs`

```rust
// Current threshold (adjust based on compare_models.py output):
const THRESHOLD: i32 = 2800;  // ADC units

// If too many false positives (normal classified as abnormal):
const THRESHOLD: i32 = 3000;  // raise it

// If too many false negatives (abnormal missed):
const THRESHOLD: i32 = 2600;  // lower it
```

After changing: `cargo build --release` and re-run the evaluation.

**When to use:** Disagreement rate is high but both models are structurally
similar — the boundary just needs shifting.

---

## Section 2 — Model Architecture Comparison

Test different MLP architectures for the tradeoff between accuracy and cost:

| Architecture | Params | Est. Inference | Expected Accuracy |
|--------------|--------|---------------|-------------------|
| Threshold (current) | 0 | ~5 µs | ~62% |
| 5 → 4 → 1 | 29 | ~15 µs | ~75% |
| 5 → 8 → 1 (current MLP) | 57 | ~25 µs | ~85% |
| 5 → 16 → 1 | 113 | ~50 µs | ~88% |
| 5 → 8 → 8 → 1 | 121 | ~70 µs | ~90% |

### How to test a different architecture

**1. Change `train_simple_model.py`:**
```python
# Edit the hidden layer sizes:
model = MLPClassifier(hidden_layer_sizes=(8,), ...)   # 5→8→1
model = MLPClassifier(hidden_layer_sizes=(16,), ...)  # 5→16→1
model = MLPClassifier(hidden_layer_sizes=(8, 8), ...) # 5→8→8→1
```

**2. Re-export and reflash:**
```powershell
py python/train_simple_model.py
py python/quantize_weights.py
py python/export_rust_weights.py   # updates model_weights.rs
```

**3. Update `inference.rs`** to match the new layer sizes.

**4. Rebuild firmware:**
```powershell
cd firmware/esp32-rust
cargo build --release
espflash flash target\xtensa-esp32-none-elf\release\ecg-esp32
```

**5. Re-run evaluation and compare.**

---

## Section 3 — Fine-Tuning Loop

Use this when quantization delta is > 5% or accuracy is consistently low on
specific pattern types.

### Concept

```
Initial model (synthetic data)
        │
        └── Evaluate on ESP32 via UART feed
                │
                └── Identify boundary/error samples
                        │
                        └── Augment training set with those samples
                                │
                                └── Retrain → re-quantize → reflash
                                        │
                                        └── Evaluate again → repeat if needed
```

### Steps

```powershell
# 1. Run evaluation, save disagreement samples
py python/uart_feed_evaluator.py --port COM3
py python/compare_models.py --save-disagreements data/disagreements.csv

# 2. Fine-tune using disagreement samples
py python/fine_tune_model.py \
    --base-data data/sample_ecg.csv \
    --extra-data data/disagreements.csv \
    --epochs 100

# 3. Re-quantize and export
py python/quantize_weights.py
py python/export_rust_weights.py

# 4. Reflash and measure improvement
cargo build --release
espflash flash target\xtensa-esp32-none-elf\release\ecg-esp32
py python/uart_feed_evaluator.py --port COM3
py python/compare_models.py
```

**Expected outcome:** Quantization delta should drop from ~3% to ~1%.

---

## Section 4 — Quantization Precision

Currently the model uses **int8** (8-bit signed integers). Alternatives:

| Precision | Range | Model size | Accuracy |
|-----------|-------|------------|----------|
| float32 | ±3.4×10³⁸ | 228 bytes | 94% (baseline) |
| int8 (current) | −128 to +127 | 84 bytes | ~91% |
| int4 | −8 to +7 | ~42 bytes | ~87% |

### How to change quantization bits

**In `python/quantize_weights.py`:**
```python
# Change this line:
QUANT_BITS = 8   # current
QUANT_BITS = 4   # try this for half the size
```

**Re-export and reflash** (same steps as Section 2, step 2–4).

**Trade-off to document:**

| Config | Model bytes | Accuracy | Inference µs |
|--------|-------------|----------|--------------|
| float32 PC | 228 | 94% | ~12 µs (PC) |
| int8 ESP32 | 84 | 91% | ~25 µs |
| int4 ESP32 | 42 | 87% | ~20 µs |

This table is the **core deliverable** of the optimization analysis.

---

## Section 5 — Signal Processing Parameters

These affect how raw ADC values are transformed before inference.

### Window Size (`RING_BUF_SIZE` in `main.rs`)

```rust
const RING_BUF_SIZE: usize = 64;   // 256 ms window — faster detection, less context
const RING_BUF_SIZE: usize = 128;  // 512 ms window — current (balanced)
const RING_BUF_SIZE: usize = 256;  // 1024 ms window — more context, higher latency
```

| Window | RAM bytes | Alert latency | Detection context |
|--------|-----------|---------------|-------------------|
| 64 | 256 B | ~256 ms | Low |
| 128 | 512 B | ~512 ms | Medium (current) |
| 256 | 1024 B | ~1024 ms | High |

### Filter Window (`FILTER_WINDOW` in `main.rs`)

```rust
const FILTER_WINDOW: usize = 4;   // less smoothing, faster response
const FILTER_WINDOW: usize = 8;   // current
const FILTER_WINDOW: usize = 16;  // more smoothing, 16 ms lag
```

### Sampling Rate (`SAMPLE_RATE_HZ` in `main.rs`)

```rust
const SAMPLE_RATE_HZ: u32 = 125;  // lower UART load, less feature resolution
const SAMPLE_RATE_HZ: u32 = 250;  // current (standard ECG)
const SAMPLE_RATE_HZ: u32 = 500;  // better QRS detection, 2× UART load
```

> **Note:** 500 Hz with 115200 baud UART may lose samples.
> Use 921600 baud or reduce UART logging to every 2nd sample.

---

## Section 6 — Evaluation Metrics Reference

Use these to decide whether an optimization is worth making:

| Metric | Formula | Good | Acceptable |
|--------|---------|------|------------|
| Accuracy | TP+TN / total | > 90% | > 85% |
| Precision | TP / (TP+FP) | > 85% | > 75% |
| Recall | TP / (TP+FN) | > 85% | > 75% |
| F1 Score | 2·P·R / (P+R) | > 85% | > 75% |
| Quant. delta | PC acc − ESP32 acc | < 3% | < 5% |
| Inference time | µs per window | < 100 µs | < 400 µs |
| Alert latency | ms from onset | < 5 ms | < 20 ms |
| Model size | bytes in flash | < 256 B | < 1 KB |
| RAM usage | static bytes | < 2 KB | < 8 KB |

---

## Recommended Optimization Sequence for a Report

```
1. Baseline
   Run evaluation → record all metrics → this is your "before" table

2. Threshold tuning (Section 1)
   Quick, no retraining — shows sensitivity analysis

3. Architecture comparison (Section 2)
   Compare 3 architectures — core table for the report

4. Quantization precision (Section 4)
   float32 vs int8 vs int4 — accuracy/size tradeoff

5. Fine-tuning (Section 3) — optional
   Iterative improvement — shows the optimization loop working

6. Final evaluation
   Re-run all metrics → "after" table → compute improvement delta
```

This gives you **5 experiments** with clear metrics at each step,
which is sufficient for a strong embedded systems project report.

---

*This guide is for educational embedded systems research only.*
