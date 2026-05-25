# Real-Time Design Rationale

This document explains the embedded design choices and timing constraints in the current ESP32-S3 firmware.

## Fixed Sampling

The target sample rate is:

```text
250 Hz = 4 ms per sample
```

Uniform sampling matters because the windowed features assume fixed time spacing. Irregular sampling changes the meaning of energy and amplitude patterns.

Current implementation:

- ADC mode uses a delay-based loop.
- UART-feed mode blocks until the next PC-provided sample arrives.
- For production-quality timing, use a hardware timer and decouple logging from sampling.

## UART Bottleneck

At 115200 baud, a CSV line can take several milliseconds to transmit. This can exceed the 4 ms sampling period if logging every sample.

Mitigations:

1. Use UART-feed mode for controlled validation.
2. Increase baud rate for dense logging.
3. Log every Nth sample.
4. Use compact binary output instead of CSV.
5. Use DMA or buffering for non-blocking serial output.

## Ring Buffer

The system uses a fixed-size ring buffer:

```text
128 samples = 512 ms at 250 Hz
```

Benefits:

- No heap allocation.
- O(1) sample insertion.
- Latest-window inference after the initial fill.
- Deterministic memory use.

Tradeoff:

- The first valid inference is delayed until the buffer is full.
- Smaller buffers reduce latency but can reduce feature quality.

## Window-Based Inference

A single ECG sample is not enough for classification. A 128-sample window gives enough context for:

- Baseline level.
- Maximum and minimum amplitude.
- Peak-to-peak amplitude.
- Energy.

The current model is intentionally simple and uses only aggregate window features.

## Inference Cost

The int8 MLP is tiny:

```text
5 inputs -> 8 hidden -> 1 output
```

Approximate operations:

- Layer 1: 40 multiply-accumulates.
- Layer 2: 8 multiply-accumulates.
- Feature extraction over 128 samples.

This is comfortably below the 4 ms sampling budget on ESP32-S3.

## Memory Use

Approximate static working memory:

| Item | Size |
|---|---:|
| Ring buffer `[i32; 128]` | 512 B |
| Moving average state `[i32; 8]` | 32 B |
| MLP weights and biases | ~84 B plus metadata |
| Stack and locals | A few KB |

No heap allocation is required in firmware.

## Rust and `no_std`

The firmware uses Embedded Rust with `no_std` because:

- It avoids a heavyweight runtime.
- Static memory use is easier to reason about.
- Rust gives bounds checks and type safety.
- Const generics make fixed buffers explicit.

The design keeps the hot path simple: integer arithmetic, fixed arrays, and predictable control flow.

## Alert Latency

Latency components:

| Component | Approximate value |
|---|---:|
| Initial window fill | 512 ms |
| Feature extraction | Microseconds |
| MLP inference | Tens of microseconds |
| GPIO toggle | Microseconds |

The window dominates latency. Optimizing the MLP from 50 us to 20 us matters less than changing the window length from 128 to 64 samples.

## Design Tradeoffs

| Choice | Benefit | Cost |
|---|---|---|
| 250 Hz sampling | Common ECG educational rate | More UART traffic than 125 Hz |
| 128-sample window | Good context | 512 ms first-window delay |
| 8-sample moving average | Simple smoothing | Small lag |
| Int8 MLP | Tiny model and fast inference | Less expressive than larger models |
| CSV logging | Easy debugging | Slow at low baud |

