//! uart_feed.rs
//! =============
//! UART Feed Mode for ESP32-S3.
//!
//! This module reads ADC values from UART instead of the ADC peripheral.
//! It uses esp-hal's UART driver, NOT direct ESP32 memory-mapped registers.
//!
//! Protocol:
//!   PC        → ESP32-S3 : "{adc_value}\n"   example: "2048\n"
//!   ESP32-S3 → PC        : "{prediction}\n"  example: "-1\n", "0\n", "1\n"
//!
//! Prediction meaning:
//!   -1 = buffer not full yet
//!    0 = normal
//!    1 = abnormal
//!
//! Common ESP32-S3 UART0 pins:
//!   GPIO44 = UART0 RX
//!   GPIO43 = UART0 TX
//!
//! DISCLAIMER: Educational prototype only. Not for clinical use.

use esp_hal::{
    uart::Uart,
    Blocking,
};

const RX_BUF_SIZE: usize = 16;

/// UART Feed Reader: reads ASCII integer ADC values from UART.
///
/// This owns the UART peripheral, so avoid using esp_println! in UART-feed mode.
/// Use `send_text()` or `send_prediction()` from this struct instead.
pub struct UartFeedReader<'d> {
    uart: Uart<'d, Blocking>,
    buf: [u8; RX_BUF_SIZE],
    pos: usize,
}

impl<'d> UartFeedReader<'d> {
    /// Create a new UART feed reader.
    pub fn new(uart: Uart<'d, Blocking>) -> Self {
        Self {
            uart,
            buf: [0u8; RX_BUF_SIZE],
            pos: 0,
        }
    }

    /// Read one byte from UART RX.
    ///
    /// esp-hal UART read is blocking when the RX buffer is empty.
    fn read_byte_blocking(&mut self) -> u8 {
        let mut byte = [0u8; 1];

        loop {
            match self.uart.read(&mut byte) {
                Ok(n) if n > 0 => return byte[0],
                Ok(_) => core::hint::spin_loop(),
                Err(_) => {
                    // On RX error, keep firmware alive and wait for next byte.
                    core::hint::spin_loop();
                }
            }
        }
    }

    /// Read one ADC value from PC.
    ///
    /// Blocks until newline is received.
    ///
    /// Returns:
    ///   Some(parsed_value) on valid line
    ///   Some(2048) on parse error or buffer overflow
    ///
    /// Empty CR/LF lines are ignored.
    pub fn read_sample(&mut self) -> Option<u16> {
        loop {
            let byte = self.read_byte_blocking();

            match byte {
                b'\n' | b'\r' => {
                    if self.pos > 0 {
                        let result = self.parse_u16();
                        self.pos = 0;
                        return Some(result);
                    }

                    // Empty line, usually CRLF second byte. Ignore.
                }

                b => {
                    if self.pos < RX_BUF_SIZE - 1 {
                        self.buf[self.pos] = b;
                        self.pos += 1;
                    } else {
                        // Buffer overflow. Reset and return safe ADC midpoint.
                        self.pos = 0;
                        return Some(2048);
                    }
                }
            }
        }
    }

    /// Parse ASCII decimal integer from internal buffer.
    ///
    /// Accepts digits and ignores spaces/tabs.
    /// Clamps to 12-bit ADC range: 0..4095.
    fn parse_u16(&self) -> u16 {
        let mut result: u32 = 0;
        let mut found_digit = false;

        for &b in &self.buf[..self.pos] {
            match b {
                b'0'..=b'9' => {
                    result = result * 10 + (b - b'0') as u32;
                    found_digit = true;

                    if result > 4095 {
                        return 4095;
                    }
                }

                b' ' | b'\t' => {
                    // Skip whitespace.
                }

                _ => {
                    // Invalid character. Stop parsing.
                    break;
                }
            }
        }

        if found_digit {
            result as u16
        } else {
            2048
        }
    }

    /// Send plain text to PC.
    pub fn send_text(&mut self, text: &str) {
        self.write_bytes(text.as_bytes());
    }

    /// Send one prediction back to PC.
    ///
    /// `pred`:
    ///   -1 = buffer not full yet
    ///    0 = normal
    ///    1 = abnormal
    pub fn send_prediction(&mut self, pred: i8) {
        match pred {
            -1 => self.write_bytes(b"-1\n"),
             0 => self.write_bytes(b"0\n"),
             1 => self.write_bytes(b"1\n"),
             _ => self.write_bytes(b"-1\n"),
        }
    }

    /// Write all bytes to UART TX.
    fn write_bytes(&mut self, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            match self.uart.write(bytes) {
                Ok(0) => {
                    core::hint::spin_loop();
                }

                Ok(n) => {
                    bytes = &bytes[n..];
                }

                Err(_) => {
                    // Keep firmware alive even if one TX write fails.
                    break;
                }
            }
        }

        let _ = self.uart.flush();
    }
}