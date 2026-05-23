//! logger.rs
//! ==========
//! CSV serial logger for UART output.
//!
//! Logs one CSV-formatted line per ECG sample over UART0 (USB serial).
//! The output can be captured by a PC for GNUPlot visualization.
//!
//! # CSV Format:
//! ```
//! time_ms,adc_value,filtered_value,inference_us,prediction,alert,alert_latency_ms
//! ```
//!
//! | Field             | Type  | Description                              |
//! |-------------------|-------|------------------------------------------|
//! | time_ms           | u32   | Timestamp in milliseconds since boot     |
//! | adc_value         | i32   | Raw ADC reading (0–4095)                 |
//! | filtered_value    | i32   | After moving average filter              |
//! | inference_us      | u32   | Inference duration in microseconds       |
//! | prediction        | u8    | 0=Normal, 1=Abnormal                     |
//! | alert             | u8    | 0=Alert Off, 1=Alert On                  |
//! | alert_latency_ms  | u32   | Time from abnormal to alert (ms)         |
//!
//! # UART Settings:
//! Baud rate: 115200 (standard for Arduino/ESP32 serial monitor)
//! Data bits: 8
//! Stop bits: 1
//! Parity: None
//!
//! # Implementation Notes:
//! We use `esp_println::println!` for UART output.
//! esp-println routes through UART0 automatically at the configured baud rate.
//! This is the simplest approach for no_std output on ESP32.
//!
//! In a production system, consider:
//!   - Non-blocking UART writes using DMA
//!   - A ring buffer for serial output to avoid blocking the sampling loop
//!   - Conditional logging (log only every Nth sample to reduce UART load)
//!
//! # UART load estimation:
//! Each CSV line is approximately 60 characters.
//! At 115200 baud ≈ 11520 bytes/s, one line takes ~5.2 ms.
//! At 250 Hz sampling (4 ms per sample), UART is the bottleneck!
//!
//! SOLUTIONS:
//! 1. Reduce baud rate to check impact vs. increase to 921600.
//! 2. Log every Nth sample (e.g., N=2 → 125 Hz effective log rate).
//! 3. Use a higher baud rate (921600 baud → ~80K bytes/s → ~0.67 ms/line).
//! 4. Use a compact binary format instead of ASCII CSV.
//!
//! For this educational demo, we use 115200 and log every sample.
//! In practice, set UART to 921600 or log every 2nd sample.

use esp_println::println;

/// Log one ECG sample as a CSV line over UART.
///
/// This function is called once per sampling interval (4 ms at 250 Hz).
///
/// # Arguments
/// * `time_ms`          - Timestamp in milliseconds
/// * `adc_value`        - Raw ADC sample (0–4095)
/// * `filtered_value`   - Filtered ADC value
/// * `inference_us`     - Inference time in microseconds (0 if no inference ran)
/// * `prediction`       - Classifier output: 0=Normal, 1=Abnormal
/// * `alert`            - Alert state: 0=Off, 1=On
/// * `alert_latency_ms` - Alert latency in milliseconds (0 if no event)
///
/// # Output format example:
/// ```
/// 1000,2048,2051,25,0,0,0
/// 1004,2060,2053,25,0,0,0
/// 1512,3800,3210,26,1,1,0
/// ```
#[inline]
pub fn log_csv(
    time_ms:          u32,
    adc_value:        i32,
    filtered_value:   i32,
    inference_us:     u32,
    prediction:       u8,
    alert:            u8,
    alert_latency_ms: u32,
) {
    // Use esp_println's println! for UART output.
    // This is blocking (waits until UART TX FIFO has space).
    // Format: time_ms,adc_value,filtered_value,inference_us,prediction,alert,alert_latency_ms
    println!(
        "{},{},{},{},{},{},{}",
        time_ms,
        adc_value,
        filtered_value,
        inference_us,
        prediction,
        alert,
        alert_latency_ms,
    );
}

/// Log only every Nth sample to reduce UART bandwidth load.
///
/// # Arguments
/// * `sample_count`     - Current sample index (monotonically increasing)
/// * `log_every_n`      - Log interval (e.g., 2 = log every 2nd sample)
/// * All other arguments: same as log_csv()
///
/// Use this when UART logging is too slow for the sampling rate.
#[inline]
pub fn log_csv_throttled(
    sample_count:     u32,
    log_every_n:      u32,
    time_ms:          u32,
    adc_value:        i32,
    filtered_value:   i32,
    inference_us:     u32,
    prediction:       u8,
    alert:            u8,
    alert_latency_ms: u32,
) {
    if log_every_n > 0 && sample_count % log_every_n == 0 {
        log_csv(
            time_ms, adc_value, filtered_value,
            inference_us, prediction, alert, alert_latency_ms,
        );
    }
}

/// Print the CSV header line.
///
/// Call this once at firmware startup before the main loop.
pub fn log_header() {
    println!("time_ms,adc_value,filtered_value,inference_us,prediction,alert,alert_latency_ms");
}

/// Print a debug message (non-CSV format).
///
/// Use this for firmware startup messages, error messages, etc.
/// CSV parsers will ignore these lines if they start with a letter
/// or if the parser skips non-numeric first fields.
///
/// Prefix with "#" to mark it as a comment line for GNUPlot:
/// GNUPlot ignores lines starting with '#' by default.
#[inline]
pub fn log_debug(msg: &str) {
    println!("# {}", msg);
}
