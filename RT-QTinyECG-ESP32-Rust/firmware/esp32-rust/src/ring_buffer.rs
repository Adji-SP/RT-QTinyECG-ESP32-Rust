//! ring_buffer.rs
//! ===============
//! Fixed-size circular (ring) buffer for real-time ECG sample storage.
//!
//! This is a fundamental data structure for real-time signal processing.
//! The ring buffer holds the most recent N samples from the ADC, forming
//! a sliding window for inference.
//!
//! # Why a ring buffer?
//! - Fixed memory: no heap allocation, size known at compile time.
//! - FIFO behavior: oldest sample is dropped when buffer is full.
//! - Constant-time push: O(1) per sample regardless of window size.
//! - Safe in no_std: uses only stack memory.
//!
//! # Const generics
//! This implementation uses Rust const generics: `RingBuffer<T, N>`.
//! The buffer size N is a compile-time constant, ensuring:
//!   - Zero heap allocation
//!   - Size verified at compile time
//!   - Optimal code generation
//!
//! # Python equivalent:
//! See python/realtime_ecg_simulator.py → class RingBuffer

// Bring in core traits (no_std safe)


/// Fixed-size circular ring buffer.
///
/// # Type parameters
/// - `T`: Element type. For ADC values, use `i32`.
/// - `N`: Buffer capacity (compile-time constant).
///
/// # Example
/// ```rust
/// let mut buf: RingBuffer<i32, 128> = RingBuffer::new();
/// buf.push(1234);
/// buf.push(5678);
/// assert!(!buf.is_full()); // Not full until 128 samples pushed
/// ```
pub struct RingBuffer<T, const N: usize> {
    /// Internal fixed-size storage
    buffer: [T; N],

    /// Index where the NEXT write will happen (write head)
    head: usize,

    /// Number of valid elements currently stored
    len: usize,
}

impl<T, const N: usize> RingBuffer<T, N>
where
    T: Copy + Default,
{
    /// Create a new empty ring buffer.
    ///
    /// All elements are initialized to `T::default()`.
    /// For i32, that's 0.
    pub fn new() -> Self {
        Self {
            buffer: [T::default(); N],
            head:   0,
            len:    0,
        }
    }

    /// Push a new value into the ring buffer.
    ///
    /// If the buffer is full, the oldest element is overwritten (FIFO).
    /// The write head wraps around automatically.
    ///
    /// # Arguments
    /// * `value` - The new sample to push.
    #[inline]
    pub fn push(&mut self, value: T) {
        // Write new value at current head position
        self.buffer[self.head] = value;

        // Advance head (wrapping around at N)
        self.head = (self.head + 1) % N;

        // Track how many valid elements we have (up to N)
        if self.len < N {
            self.len += 1;
        }
        // When len == N (buffer full), subsequent pushes overwrite old data.
        // head already advanced, so next push overwrites the oldest sample.
    }

    /// Returns true if the buffer has been filled at least once.
    ///
    /// Inference should only run when the buffer is full (a complete window).
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len == N
    }

    /// Returns the number of currently stored elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the buffer contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a slice view of the buffer contents IN ORDER (oldest first).
    ///
    /// If the buffer is not full, returns a slice of the elements pushed so far.
    ///
    /// **Note**: The internal storage is a circular array, so this method
    /// reconstructs the logical order by returning from the current "start"
    /// position. This is an in-place slice only when the buffer hasn't wrapped.
    ///
    /// For the inference window, we return a slice of all N samples in
    /// insertion order (oldest at index 0, newest at index N-1).
    ///
    /// # Returns
    /// Slice of stored elements in chronological order.
    ///
    /// **⚠️ Limitation**: This implementation returns the raw internal buffer slice.
    /// When the buffer has wrapped, the samples are NOT in chronological order.
    /// For a fully correct implementation, copy to a temp array.
    ///
    /// For the simple feature extraction (mean, max, min, peak-to-peak, energy),
    /// order doesn't matter, so this is safe for our inference.
    pub fn as_slice(&self) -> &[T] {
        // Return the full internal buffer slice.
        // When full, all N elements are valid (though order may be circular).
        // For order-independent features (mean, max, etc.), this is correct.
        &self.buffer[..self.len]
    }

    /// Copy all current samples into a scratch buffer in chronological order.
    ///
    /// This is the correct version when order matters (e.g., for peak detection).
    ///
    /// # Arguments
    /// * `out` - Output buffer of at least N elements. Values are written
    ///           from index 0 (oldest) to index N-1 (newest).
    pub fn copy_to_ordered(&self, out: &mut [T]) {
        let n = self.len.min(out.len());
        if !self.is_full() {
            // Buffer hasn't wrapped: elements are in order starting at index 0
            out[..n].copy_from_slice(&self.buffer[..n]);
        } else {
            // Buffer has wrapped: oldest element is at current head position
            let oldest = self.head; // head points to where next write goes = oldest
            for i in 0..n {
                out[i] = self.buffer[(oldest + i) % N];
            }
        }
    }

    /// Clear the ring buffer (reset to empty state).
    pub fn clear(&mut self) {
        self.head = 0;
        self.len  = 0;
        // Note: doesn't zero the buffer contents (not needed for correctness)
    }
}

// ── Feature methods (specific to i32) ────────────────────────────────────────

impl<const N: usize> RingBuffer<i32, N> {
    /// Compute the arithmetic mean of all stored samples.
    ///
    /// Returns 0 if the buffer is empty.
    /// Uses integer arithmetic (no float) for embedded safety.
    ///
    /// Note: sum is computed in i64 to avoid overflow for large N.
    /// With N=128 and max ADC value 4095: 128 × 4095 = 524,160 → fits in i32.
    /// But for safety, we use i64.
    pub fn mean(&self) -> i32 {
        if self.len == 0 {
            return 0;
        }
        let sum: i64 = self.buffer[..self.len]
            .iter()
            .map(|&x| x as i64)
            .sum();
        (sum / self.len as i64) as i32
    }

    /// Find the maximum value in the buffer.
    ///
    /// Returns i32::MIN if buffer is empty.
    pub fn max_val(&self) -> i32 {
        self.buffer[..self.len]
            .iter()
            .copied()
            .max()
            .unwrap_or(i32::MIN)
    }

    /// Find the minimum value in the buffer.
    ///
    /// Returns i32::MAX if buffer is empty.
    pub fn min_val(&self) -> i32 {
        self.buffer[..self.len]
            .iter()
            .copied()
            .min()
            .unwrap_or(i32::MAX)
    }

    /// Compute peak-to-peak amplitude (max - min).
    ///
    /// This is a key feature for arrhythmia detection:
    /// - Normal ECG: moderate peak-to-peak (R-peak to S-wave)
    /// - Erratic signal: very high peak-to-peak
    pub fn peak_to_peak(&self) -> i32 {
        if self.len == 0 {
            return 0;
        }
        self.max_val() - self.min_val()
    }

    /// Compute signal energy (sum of squares, normalized by N).
    ///
    /// Uses i64 intermediate to prevent overflow.
    /// Result is divided by N to normalize across window size.
    pub fn energy(&self) -> i64 {
        if self.len == 0 {
            return 0;
        }
        let sum_sq: i64 = self.buffer[..self.len]
            .iter()
            .map(|&x| (x as i64) * (x as i64))
            .sum();
        sum_sq / self.len as i64
    }
}

// ── Default implementation ────────────────────────────────────────────────────

impl<T, const N: usize> Default for RingBuffer<T, N>
where
    T: Copy + Default,
{
    fn default() -> Self {
        Self::new()
    }
}
