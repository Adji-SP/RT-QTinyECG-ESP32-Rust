//! filter.rs
//! ==========
//! Signal filtering utilities for ECG preprocessing.
//!
//! Filters implemented here match the Python equivalents in:
//!   python/preprocessing.py
//!
//! All filters are implemented using only integer arithmetic where possible,
//! with no heap allocation, for embedded compatibility.
//!
//! # Filters:
//! - `MovingAverageState`: Causal moving average filter (real-time safe)
//! - `moving_average_window()`: Compute moving average over a slice
//! - `baseline_remove()`: Remove baseline using long moving average
//! - `normalize_adc_to_i16()`: Map ADC value to [-32768, +32767]
//!
//! # Why moving average?
//! - Extremely simple and fast (O(1) per sample with state machine)
//! - Attenuates high-frequency noise above ~30 Hz (at 250 Hz, window=8)
//! - No floating point required for integer approximation
//! - Deterministic execution time (important for real-time loops)
//!
//! # Python equivalent:
//! See python/preprocessing.py → moving_average(), baseline_remove()

// ─── Moving Average State Machine ────────────────────────────────────────────

/// Causal moving average filter with fixed window size.
///
/// Maintains a circular state buffer of the last N samples.
/// Each call to `push_and_average()` adds one sample and returns
/// the average of the last N samples.
///
/// This is the real-time version of moving average:
///   - No look-ahead (causal)
///   - O(1) per sample
///   - No division operation (replaced by right-shift if N is power of 2)
///
/// # Type parameter
/// - `N`: Filter window size (compile-time constant, must be > 0)
///
/// # Example
/// ```rust
/// let mut filt = MovingAverageState::<8>::new();
/// let avg = filt.push_and_average(2048);  // returns 2048 (only 1 sample)
/// let avg = filt.push_and_average(2100);  // returns average of 2 samples
/// ```
pub struct MovingAverageState<const N: usize> {
    /// Circular state buffer for recent samples
    samples: [i32; N],
    /// Write index (points to next write position)
    head: usize,
    /// Number of valid samples (0..=N)
    count: usize,
    /// Running sum of all samples in the window (for O(1) update)
    running_sum: i64,
}

impl<const N: usize> MovingAverageState<N> {
    /// Create a new moving average filter with all-zero initial state.
    pub fn new() -> Self {
        assert!(N > 0, "Filter window size must be > 0");
        Self {
            samples:     [0i32; N],
            head:        0,
            count:       0,
            running_sum: 0,
        }
    }

    /// Push a new sample and return the current moving average.
    ///
    /// Uses a running sum for O(1) computation:
    ///   running_sum = running_sum + new_sample - oldest_sample
    ///   average     = running_sum / count
    ///
    /// # Arguments
    /// * `sample` - New ADC value (i32, typically 0–4095)
    ///
    /// # Returns
    /// Current moving average as i32.
    #[inline]
    pub fn push_and_average(&mut self, sample: i32) -> i32 {
        // Subtract the oldest sample from the running sum (will be overwritten)
        let old_sample = self.samples[self.head];
        self.running_sum -= old_sample as i64;

        // Write new sample
        self.samples[self.head] = sample;
        self.running_sum += sample as i64;

        // Advance head
        self.head = (self.head + 1) % N;

        // Track count (up to N)
        if self.count < N {
            self.count += 1;
        }

        // Compute average
        // Integer division: truncates toward zero.
        // For embedded: if N is a power of 2, could use right shift instead.
        if self.count == 0 {
            0
        } else {
            (self.running_sum / self.count as i64) as i32
        }
    }

    /// Reset the filter to zero initial state.
    pub fn reset(&mut self) {
        self.samples     = [0i32; N];
        self.head        = 0;
        self.count       = 0;
        self.running_sum = 0;
    }

    /// Return the current moving average without pushing a new sample.
    pub fn current_average(&self) -> i32 {
        if self.count == 0 {
            0
        } else {
            (self.running_sum / self.count as i64) as i32
        }
    }
}

impl<const N: usize> Default for MovingAverageState<N> {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Windowed Moving Average (over a slice) ───────────────────────────────────

/// Compute the moving average of a slice without state.
///
/// For each position i, computes the average of samples[i-window+1..=i].
/// For positions before the first full window, uses all available samples.
///
/// This is used for feature extraction over a complete inference window,
/// not for the per-sample real-time path.
///
/// # Arguments
/// * `samples` - Slice of ADC samples (e.g., ring buffer content)
/// * `window`  - Window size in samples
/// * `output`  - Output buffer (must be same length as samples)
///
/// # Panics
/// Panics if `output.len() < samples.len()`.
pub fn moving_average_window(samples: &[i32], window: usize, output: &mut [i32]) {
    assert!(output.len() >= samples.len());
    let n = samples.len();
    for i in 0..n {
        let start: usize = if i + 1 >= window { i + 1 - window } else { 0 };
        let count = i + 1 - start;
        let sum: i64 = samples[start..=i]
            .iter()
            .map(|&x| x as i64)
            .sum();
        output[i] = (sum / count as i64) as i32;
    }
}

// ─── Baseline Removal ─────────────────────────────────────────────────────────

/// Remove signal baseline from a window of samples.
///
/// Baseline is estimated as the long moving average (window = `baseline_window`).
/// Subtracting baseline isolates the AC component (the actual ECG beats).
///
/// # Arguments
/// * `samples`         - Input slice of ADC values
/// * `baseline_window` - Long window for baseline (e.g., 64 at 250 Hz = 256 ms)
/// * `output`          - Output slice for baseline-removed values
///                       (same length as samples, may contain negative values)
/// * `scratch`         - Scratch buffer for intermediate baseline computation
///                       (must be >= samples.len())
pub fn baseline_remove(
    samples: &[i32],
    baseline_window: usize,
    output: &mut [i32],
    scratch: &mut [i32],
) {
    let n = samples.len();
    assert!(output.len() >= n);
    assert!(scratch.len() >= n);

    // Compute baseline as long moving average
    moving_average_window(samples, baseline_window, scratch);

    // Subtract baseline from signal
    for i in 0..n {
        output[i] = samples[i] - scratch[i];
    }
}

// ─── ADC Normalization ────────────────────────────────────────────────────────

/// Normalize a single ADC value from [0, 4095] to i16 range [-32768, +32767].
///
/// This is used to convert ADC values into a format suitable for
/// integer arithmetic in the quantized MLP inference.
///
/// Mapping:
///   0    → -32768
///   2047 → ~0
///   4095 → +32767
///
/// Formula:
///   normalized = (value * 65536 / 4095) - 32768
///   (all integer arithmetic)
///
/// # Arguments
/// * `adc_value` - Raw ADC value (0–4095)
///
/// # Returns
/// Normalized value in range [-32768, +32767] as i16.
///
/// # Python equivalent:
/// python/preprocessing.py → normalize_adc_to_i16()
#[inline]
pub fn normalize_adc_to_i16(adc_value: i32) -> i16 {
    // Scale from [0, 4095] to [0, 65535]
    // Then shift to [-32768, 32767]
    // Use i64 for intermediate to avoid overflow: 4095 * 65536 = 268,369,920 (fits in i64)
    let scaled: i64 = (adc_value as i64 * 65536i64) / 4095i64;
    let shifted: i64 = scaled - 32768i64;
    // Clamp to i16 range
    shifted.max(-32768).min(32767) as i16
}

/// Normalize an entire window of ADC values in-place.
///
/// # Arguments
/// * `window` - Slice of i32 ADC values; each value is normalized to i16 range
///              and written back (sign-extended back to i32 for use in inference).
pub fn normalize_window(window: &mut [i32]) {
    for v in window.iter_mut() {
        *v = normalize_adc_to_i16(*v) as i32;
    }
}

// ─── Simple Clamping Helper ───────────────────────────────────────────────────

/// Clamp a value to the valid ADC range [0, 4095].
#[inline]
pub fn clamp_adc(val: i32) -> i32 {
    val.max(0).min(4095)
}
