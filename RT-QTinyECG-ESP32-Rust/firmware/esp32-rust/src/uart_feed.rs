//! uart_feed.rs
//! =============
//! UART Feed Mode — reads ADC values from UART RX instead of the ADC peripheral.
//!
//! Uses direct ESP32 UART0 register access (memory-mapped I/O) to avoid
//! esp-hal 1.x Uart<'d> lifetime complexity. UART0 is shared with esp-println
//! (TX only) — this module only accesses RX FIFO registers, no conflict.
//!
//! ## ESP32 UART0 Register Map (Technical Reference Manual §13)
//!   Base: 0x3FF4_0000
//!   0x00 = UART_FIFO_REG    — write TX / read RX FIFO
//!   0x1C = UART_STATUS_REG  — bits[7:0] = RXFIFO_CNT (bytes waiting)
//!
//! ## Protocol
//!   PC  →  ESP32: "{adc_value}\n"   e.g. "2048\n"
//!   ESP32 → PC  : "{prediction}\n" e.g. "0\n", "1\n", "-1\n"
//!
//! ## Hardware (ESP32 DevKit V1)
//!   GPIO3 = UART0 RX ← PC (via USB-UART chip, same USB cable)
//!   GPIO1 = UART0 TX → PC (used by esp-println, no conflict)
//!
//! DISCLAIMER: Educational prototype only. Not for clinical use.

// UART0 memory-mapped register addresses on ESP32
const UART0_BASE:       u32 = 0x3FF4_0000;
const UART_FIFO_REG:    u32 = UART0_BASE + 0x00;   // RX/TX FIFO
const UART_STATUS_REG:  u32 = UART0_BASE + 0x1C;   // status

// Buffer for accumulating one ASCII integer line
const RX_BUF_SIZE: usize = 16;

/// UART Feed Reader: reads ASCII integer ADC values from UART0 RX.
///
/// No HAL ownership needed — works by direct register access.
/// Zero-copy, no heap, no_std safe.
pub struct UartFeedReader {
    buf: [u8; RX_BUF_SIZE],
    pos: usize,
}

impl UartFeedReader {
    /// Create a new reader. Call once in main().
    pub const fn new() -> Self {
        Self { buf: [0u8; RX_BUF_SIZE], pos: 0 }
    }

    /// Read one byte from UART0 RX FIFO (non-blocking).
    /// Returns None if FIFO is empty.
    #[inline]
    fn read_byte_nonblocking(&self) -> Option<u8> {
        // Safety: UART0 registers are valid memory-mapped I/O on ESP32.
        // Read-only access to FIFO (RX path) while esp-println uses TX path.
        unsafe {
            let status = core::ptr::read_volatile(UART_STATUS_REG as *const u32);
            let rxfifo_cnt = status & 0xFF;
            if rxfifo_cnt > 0 {
                let byte = core::ptr::read_volatile(UART_FIFO_REG as *const u32) as u8;
                Some(byte)
            } else {
                None
            }
        }
    }

    /// Read one byte from UART0 RX FIFO (blocking — spins until data arrives).
    #[inline]
    fn read_byte_blocking(&self) -> u8 {
        loop {
            if let Some(b) = self.read_byte_nonblocking() {
                return b;
            }
            // Spin wait — ~4 ns per iteration @ 240 MHz.
            // Could use WFI (wait-for-interrupt) for power savings,
            // but that adds complexity for this educational prototype.
            core::hint::spin_loop();
        }
    }

    /// Read one ADC value from PC (blocks until newline).
    ///
    /// Returns parsed u16, or 2048 (ADC midpoint) on error.
    /// Returns None on empty line (CRLF pair or keep-alive).
    pub fn read_sample(&mut self) -> Option<u16> {
        loop {
            let byte = self.read_byte_blocking();

            match byte {
                // Line terminator: parse accumulated buffer
                b'\n' | b'\r' => {
                    if self.pos > 0 {
                        let result = self.parse_u16();
                        self.pos = 0;
                        return Some(result);
                    }
                    // Empty line → ignore (CRLF second byte or keep-alive)
                }
                // Accumulate digit byte into buffer
                b => {
                    if self.pos < RX_BUF_SIZE - 1 {
                        self.buf[self.pos] = b;
                        self.pos += 1;
                    } else {
                        // Buffer overflow → reset, return safe midpoint
                        self.pos = 0;
                        return Some(2048);
                    }
                }
            }
        }
    }

    /// Parse ASCII decimal integer from internal buffer.
    /// Handles optional leading/trailing whitespace (spaces, tabs).
    #[inline]
    fn parse_u16(&self) -> u16 {
        let mut result: u32 = 0;
        let mut found_digit = false;

        for &b in &self.buf[..self.pos] {
            match b {
                b'0'..=b'9' => {
                    result = result * 10 + (b - b'0') as u32;
                    found_digit = true;
                    if result > 4095 {
                        return 4095; // clamp to 12-bit ADC max
                    }
                }
                b' ' | b'\t' => {
                    // skip whitespace
                }
                _ => {
                    // invalid character → stop parsing
                    break;
                }
            }
        }

        if found_digit { result as u16 } else { 2048 }
    }

    /// Send prediction back to PC via UART0 TX (println! macro).
    ///
    /// `pred` = 0 (Normal), 1 (Abnormal), or -1 (buffer not yet full).
    pub fn send_prediction(&self, pred: i8) {
        // esp_println uses UART0 TX independently from our RX reads.
        esp_println::println!("{}", pred);
    }
}
