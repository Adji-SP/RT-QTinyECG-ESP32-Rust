//! main.rs
//! ========
//! RT-QTinyECG-ESP32-Rust — ESP32 no_std firmware (esp-hal 1.x)
//!
//! ## Build Modes
//!
//! ### ADC Mode (default — real sensor)
//! ```powershell
//! cargo build --release
//! ```
//! Reads ECG from AD8232 sensor via ADC on GPIO34.
//!
//! ### UART Feed Mode (simulation — no sensor needed)
//! ```powershell
//! cargo build --release --features uart-feed
//! ```
//! Receives ADC values from PC via UART RX (GPIO3).
//! PC sends "2048\n" → ESP32 responds "0\n" or "1\n".
//! Use with: py python/uart_feed_evaluator.py --port COM3
//!
//! ## Pipeline (250 Hz / 4 ms loop)
//!   ADC/UART → moving avg filter → ring buffer → inference → LED/Buzzer → UART log
//!
//! Target : xtensa-esp32-none-elf, Xtensa Rust 1.95+
//! DISCLAIMER: Educational prototype only. Not a medical device.

#![no_std]
#![no_main]

// ── Pure Rust modules (no HAL dependencies) ────────────────────────────────────
mod ring_buffer;
mod filter;
mod inference;
mod model_weights;
mod logger;

// ── UART Feed Mode module (only compiled with --features uart-feed) ─────────────
#[cfg(feature = "uart-feed")]
mod uart_feed;

// ── esp-hal 1.1 imports ───────────────────────────────────────────────────────
// ADC + Delay + GPIO are in the "unstable" feature group in esp-hal 1.1.
use esp_hal::{
    delay::Delay,
    gpio::{Level, Output, OutputConfig},
};

// ADC imports: only needed in ADC mode (not uart-feed mode)
#[cfg(not(feature = "uart-feed"))]
use esp_hal::analog::adc::{Adc, AdcConfig, Attenuation};

// In esp-hal 1.1.1, #[entry] is provided by xtensa-lx-rt (v0.22).
use xtensa_lx_rt::entry;
use esp_backtrace as _;

// ── Constants ─────────────────────────────────────────────────────────────────
/// Sampling rate in Hz (250 Hz = 4 ms period)
const SAMPLE_RATE_HZ:     u32   = 250;
/// Sampling interval in microseconds
const SAMPLE_INTERVAL_US: u32   = 1_000_000 / SAMPLE_RATE_HZ; // 4000 µs
/// Ring buffer size in samples (128 @ 250 Hz = 512 ms window)
const RING_BUF_SIZE:      usize = 128;
/// Moving average filter window (8 samples)
const FILTER_WINDOW:      usize = 8;

// ── Entry point ───────────────────────────────────────────────────────────────
#[entry]
fn main() -> ! {
    // ── 1. Initialize all ESP32 peripherals ────────────────────────────────────
    // esp-hal 1.x: one call replaces Peripherals::take() + SystemControl + ClockControl.
    // Config::default() sets CPU to maximum frequency (240 MHz on ESP32).
    let peripherals = esp_hal::init(esp_hal::Config::default());

    // ── 2. GPIO: LED (GPIO2) and Buzzer (GPIO25) ───────────────────────────────
    // esp-hal 1.1: Output::new() takes 3 args: (pin, level, OutputConfig)
    let mut led    = Output::new(peripherals.GPIO2,  Level::Low, OutputConfig::default());
    let mut buzzer = Output::new(peripherals.GPIO25, Level::Low, OutputConfig::default());

    // ── 3A. ADC Mode: ECG input from GPIO34 (AD8232 sensor) ───────────────────
    #[cfg(not(feature = "uart-feed"))]
    let (mut adc, mut ecg_pin) = {
        let mut adc_config = AdcConfig::new();
        let pin = adc_config.enable_pin(peripherals.GPIO34, Attenuation::_11dB);
        let adc = Adc::new(peripherals.ADC1, adc_config);
        (adc, pin)
    };

    // ── 3B. UART Feed Mode: receive ADC values from PC over UART RX (GPIO3) ────
    // UartFeedReader uses direct register access — no peripheral ownership needed.
    #[cfg(feature = "uart-feed")]
    let mut uart_reader = uart_feed::UartFeedReader::new();

    // ── 4. Delay ───────────────────────────────────────────────────────────────
    // esp-hal 1.x: Delay::new() takes no arguments.
    let delay = Delay::new();

    // ── 5. Signal processing state ─────────────────────────────────────────────
    let mut ring_buf = ring_buffer::RingBuffer::<i32, RING_BUF_SIZE>::new();
    let mut filt     = filter::MovingAverageState::<FILTER_WINDOW>::new();

    // ── 6. Application state variables ────────────────────────────────────────
    let mut time_ms:      u32  = 0;
    let mut alert_active: bool = false;

    // ── 7. UART CSV header ─────────────────────────────────────────────────────
    #[cfg(not(feature = "uart-feed"))]
    logger::log_header();

    #[cfg(feature = "uart-feed")]
    esp_println::println!("# UART_FEED_MODE ready. Send ADC values (e.g. '2048\\n').");

    // ── 8. Main loop ───────────────────────────────────────────────────────────
    loop {
        // ── a) Get next ADC sample ─────────────────────────────────────────────
        // ADC Mode: wait 4 ms then read from GPIO34
        #[cfg(not(feature = "uart-feed"))]
        let adc_raw: u16 = {
            delay.delay_micros(SAMPLE_INTERVAL_US);
            nb::block!(adc.read_oneshot(&mut ecg_pin)).unwrap_or(2048)
        };

        // UART Feed Mode: block until PC sends a value (no hardware delay needed)
        #[cfg(feature = "uart-feed")]
        let adc_raw: u16 = uart_reader.read_sample().unwrap_or(2048);

        // ── b) Moving average filter (O(1), mirrors Python preprocessing) ────────
        let filtered: i32 = filt.push_and_average(adc_raw as i32);

        // ── c) Push into ring buffer (sliding window) ─────────────────────────────
        ring_buf.push(filtered);

        // Default per-loop outputs
        let mut prediction:       u8  = 0;
        let mut inference_us:     u32 = 0;
        let mut alert_latency_ms: u32 = 0;

        // ── d) Inference (runs every sample once buffer is full) ──────────────────
        if ring_buf.is_full() {
            let window: &[i32] = ring_buf.as_slice();

            // Run int8 threshold or MLP classifier (see inference.rs)
            prediction   = inference::infer(window);
            // Placeholder: ~25 µs for threshold @ 240 MHz
            // Real measurement: use xtensa_lx::timer::get_cycle_count()
            inference_us = 25;

            // ── e) Alert state machine ────────────────────────────────────────────
            if prediction == 1 {
                if !alert_active {
                    alert_active     = true;
                    alert_latency_ms = 0;
                }
                led.set_high();
                buzzer.set_high();
            } else {
                if alert_active { alert_active = false; }
                led.set_low();
                buzzer.set_low();
                alert_latency_ms = 0;
            }
        }

        // ── f) Output: depends on build mode ─────────────────────────────────────

        // ADC Mode: log full CSV line over UART (for post-analysis)
        #[cfg(not(feature = "uart-feed"))]
        logger::log_csv(
            time_ms,
            adc_raw as i32,
            filtered,
            inference_us,
            prediction,
            alert_active as u8,
            alert_latency_ms,
        );

        // UART Feed Mode: send only the prediction back to PC (for real-time comparison)
        // Protocol: "0\n" = Normal, "1\n" = Abnormal, "-1\n" = buffer not full
        #[cfg(feature = "uart-feed")]
        {
            let pred_out: i8 = if ring_buf.is_full() { prediction as i8 } else { -1 };
            uart_reader.send_prediction(pred_out);
        }

        // ── g) Advance timestamp ───────────────────────────────────────────────────
        time_ms = time_ms.wrapping_add(SAMPLE_INTERVAL_US / 1000);
    }
}
