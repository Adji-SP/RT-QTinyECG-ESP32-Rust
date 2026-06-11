//! ring_buffer.rs
//! ===============
//! Fixed-size circular (ring) buffer — local test version.
//!
//! Direct port of firmware/esp32-rust/src/ring_buffer.rs adapted for std.

#![allow(dead_code)]

//! Direct port of firmware/esp32-rust/src/ring_buffer.rs adapted for std.

/// Fixed-size circular ring buffer.
pub struct RingBuffer<T, const N: usize> {
    buffer: [T; N],
    head:   usize,
    len:    usize,
}

impl<T, const N: usize> RingBuffer<T, N>
where
    T: Copy + Default,
{
    pub fn new() -> Self {
        Self {
            buffer: [T::default(); N],
            head:   0,
            len:    0,
        }
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        self.buffer[self.head] = value;
        self.head = (self.head + 1) % N;
        if self.len < N {
            self.len += 1;
        }
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.len == N
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a slice of the buffer contents.
    /// For order-independent features (mean, max, etc.) this is correct.
    pub fn as_slice(&self) -> &[T] {
        &self.buffer[..self.len]
    }

    /// Copy all current samples into a scratch buffer in chronological order.
    pub fn copy_to_ordered(&self, out: &mut [T]) {
        let n = self.len.min(out.len());
        if !self.is_full() {
            out[..n].copy_from_slice(&self.buffer[..n]);
        } else {
            let oldest = self.head;
            for i in 0..n {
                out[i] = self.buffer[(oldest + i) % N];
            }
        }
    }

    pub fn clear(&mut self) {
        self.head = 0;
        self.len  = 0;
    }
}

impl<T, const N: usize> Default for RingBuffer<T, N>
where
    T: Copy + Default,
{
    fn default() -> Self {
        Self::new()
    }
}
