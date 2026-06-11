//! filter.rs
//! ==========
//! Signal filtering utilities for ECG preprocessing — local test version.
//!
//! Direct port of firmware/esp32-rust/src/filter.rs adapted for std.
//! The logic is byte-for-byte identical to the firmware version.

#![allow(dead_code)]

// ─── Moving Average State Machine ────────────────────────────────────────────

/// Causal moving average filter with fixed window size.
///
/// Maintains a circular state buffer of the last N samples.
/// Each call to `push_and_average()` adds one sample and returns
/// the average of the last N samples.
pub struct MovingAverageState<const N: usize> {
    samples: [i32; N],
    head: usize,
    count: usize,
    running_sum: i64,
}

impl<const N: usize> MovingAverageState<N> {
    pub fn new() -> Self {
        assert!(N > 0, "Filter window size must be > 0");
        Self {
            samples:     [0i32; N],
            head:        0,
            count:       0,
            running_sum: 0,
        }
    }

    #[inline]
    pub fn push_and_average(&mut self, sample: i32) -> i32 {
        let old_sample = self.samples[self.head];
        self.running_sum -= old_sample as i64;
        self.samples[self.head] = sample;
        self.running_sum += sample as i64;
        self.head = (self.head + 1) % N;
        if self.count < N {
            self.count += 1;
        }
        if self.count == 0 {
            0
        } else {
            (self.running_sum / self.count as i64) as i32
        }
    }

    pub fn reset(&mut self) {
        self.samples     = [0i32; N];
        self.head        = 0;
        self.count       = 0;
        self.running_sum = 0;
    }
}

impl<const N: usize> Default for MovingAverageState<N> {
    fn default() -> Self {
        Self::new()
    }
}
