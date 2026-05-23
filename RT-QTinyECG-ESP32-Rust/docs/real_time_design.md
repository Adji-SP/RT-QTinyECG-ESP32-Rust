# Real-Time Design Rationale

This document explains the architectural decisions made in RT-QTinyECG-ESP32-Rust
and why they matter for embedded real-time systems.

---

## 1. Why Fixed Sampling Interval Matters

**Problem:**
ECG signal processing assumes uniform time spacing between samples.
If samples are taken at irregular intervals:
- Frequency-domain analysis becomes invalid
- Feature computations (mean, energy) are biased
- Beat-to-beat interval (RR interval) measurements are wrong

**Solution:**
Use a fixed-rate sampling loop at exactly 250 Hz (4 ms per sample).

**Implementation in this project:**
- `delay_micros(4000)` provides approximate 250 Hz timing
- A hardware timer ISR would be more accurate

**Jitter tolerance:**
Signal processing algorithms typically tolerate ≤ 1% timing jitter.
At 4 ms interval, ≤ 40 µs jitter is acceptable.

**UART bottleneck:**
At 115200 baud, one 60-character CSV line takes ~5.2 ms to transmit.
This is longer than the 4 ms sampling interval!

Solutions:
1. Increase UART to 921600 baud → ~0.7 ms/line
2. Log every 2nd sample (125 Hz effective logging rate)
3. Use DMA-driven UART (non-blocking)
4. Use binary protocol instead of ASCII

---

## 2. Why a Ring Buffer is Used

**Problem:**
We need to process a window of recent samples without waiting to collect them
all before starting to accept new ones. A naive approach would collect 128
samples, process them, then collect 128 more (no overlap, high latency).

**Ring buffer advantages:**
1. **Sliding window**: Inference runs on the latest 128 samples at all times
2. **Fixed memory**: No dynamic allocation, no fragmentation
3. **O(1) push**: Adding a sample is constant time regardless of window size
4. **Overlap**: New samples are available immediately after the oldest is dropped

**Alternative: batch collection:**
Collect 128 samples, process, then collect 128 more.
- Pro: Simpler logic
- Con: First inference delayed by 128 × 4 ms = 512 ms, no overlap

**Ring buffer latency:**
The sliding window means inference always uses the most recent 128 samples.
After the initial fill (512 ms), each new sample triggers a potential new inference.

---

## 3. Why Inference Must Be Window-Based

**Problem:**
A single ECG sample carries very little information:
- One ADC reading could be 2050 or 2100 — both are "normal"
- A single spike could be a heartbeat R-peak OR a motion artifact
- You cannot distinguish Normal from Abnormal from one sample

**Window-based inference:**
By analyzing a window of 128 samples (512 ms), we observe:
- Multiple heartbeat cycles (at 72 BPM: ~62 ms/beat → ~8 beats in 512 ms)
- Baseline trends (slow changes visible over 512 ms)
- Amplitude patterns (sustained high amplitude vs. single spike)

**Feature window size tradeoff:**

| Window Size | Duration @ 250 Hz | Captures | Latency |
|---|---|---|---|
| 32 | 128 ms | < 1 beat | Low |
| 64 | 256 ms | ~1 beat | Medium |
| 128 | 512 ms | ~4–8 beats | Medium-High |
| 256 | 1024 ms | ~8–16 beats | High |

This project uses **128 samples (512 ms)** as a balance between
latency and feature quality.

---

## 4. Why Alert Latency Should Be Measured

**In embedded safety systems, latency matters:**
- If a cardiac abnormality is detected at time T...
- ...and the alert fires at time T + 600 ms...
- ...the system is delayed by 600 ms

**Breakdown of latency in this system:**

| Component | Latency | Reducible? |
|---|---|---|
| Window fill (128 samples) | 512 ms | Yes (smaller window) |
| Feature extraction | ~5 µs | Minimal |
| Classifier inference | ~5–50 µs | Minimal |
| GPIO toggle | ~2 µs | No |
| Total | ~512 ms | Partially |

**Key insight:**
The dominant latency is the window fill time, not the algorithm.
Optimizing inference speed from 50 µs to 5 µs saves only 45 µs
against a 512 ms total latency — a 0.009% improvement.

To truly reduce alert latency:
1. Use a smaller inference window (64 samples → 256 ms total latency)
2. Add a fast pre-filter: threshold on raw ADC before waiting for full window
3. Use R-peak detection to trigger inference immediately on each beat

**Measuring latency:**
In firmware: `alert_latency_ms = time_ms_at_alert - time_ms_at_detection`
On PC simulator: simulated with `gpio_delay_ms = 1.5` (realistic GPIO write time).

---

## 5. Why Rust Helps with Memory Safety

**Embedded systems traditionally use C** for firmware due to:
- Low-level hardware access
- No OS dependency
- Minimal runtime

**Problems with C in safety-critical contexts:**
- Buffer overflows: writing past array bounds corrupts memory silently
- Use-after-free: accessing freed memory → undefined behavior
- Null pointer dereferences: crash or silent corruption
- Integer overflow: undefined behavior in C

**Rust provides memory safety guarantees at compile time:**

| Safety Issue | C | Rust |
|---|---|---|
| Buffer overflow | Runtime (undefined behavior) | Compile-time bounds check |
| Use-after-free | Runtime crash / corruption | Compiler rejects |
| Null pointer | Runtime crash | No null (use Option<T>) |
| Data races | Runtime (concurrency bug) | Compiler rejects |
| Integer overflow | Undefined behavior | Debug: panic, Release: wrapping |

**In this firmware:**
- The ring buffer uses const generics: `RingBuffer<T, N>` — size N is verified at compile time
- All slice accesses are bounds-checked in debug builds
- No `unsafe` code except where HAL requires it (hardware register access)
- No heap allocation (`no_std` + no allocator) → no use-after-free possible

**`no_std` advantages:**
- No standard library → no hidden heap allocations
- Predictable memory layout: everything is on the stack or static
- No OS scheduler interference → deterministic real-time behavior

---

## 6. Why Integer Arithmetic for Inference

**Floating point (f32/f64) on ESP32:**
- ESP32 has an FPU (Floating Point Unit) — f32 operations are fast (~1 cycle)
- However, f32 adds code complexity and potential for subtle precision errors

**Integer (i8/i32) arithmetic advantages:**
1. **Portability**: works on processors without FPU (Cortex-M0, AVR, etc.)
2. **Predictability**: integer overflow is well-defined in Rust (checked/wrapping)
3. **Model size**: int8 weights = 4× smaller than float32
4. **Speed**: on embedded processors without FPU, int8 MACs are faster than f32

**In this project:**
- Feature extraction: all integer arithmetic
- MLP inference: int8 weights, i32 accumulators
- Only scale factors (optional, for dequantization debug) use f32

---

## Design Summary

| Design Choice | Rationale |
|---|---|
| Fixed 250 Hz sampling | Consistent time basis for signal processing |
| Ring buffer (size 128) | Sliding window without memory allocation |
| Moving average (N=8) | Simple, fast, causal noise reduction |
| Window-based inference | Features require multiple samples to be meaningful |
| Threshold classifier | No weights, fast, transparent, easily tuned |
| Int8 MLP | Tiny model, no FPU dependency, 4× smaller than f32 |
| no_std | No OS, predictable timing, no heap allocation |
| Rust | Memory safety at compile time, no buffer overflows |
| UART CSV logging | Human-readable, compatible with GNUPlot, easy capture |
