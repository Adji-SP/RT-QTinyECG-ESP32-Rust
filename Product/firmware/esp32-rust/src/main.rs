//! main.rs
//! ========
//! RT-QTinyECG-ESP32-Rust — ESP32-S3 no_std firmware (esp-hal 1.x)
//!
//! ## Build Modes
//!
//! ### ADC Mode, default real sensor
//! ```powershell
//! cargo build --release --target xtensa-esp32s3-none-elf
//! ```
//! Reads ECG from AD8232 sensor via ADC on GPIO4.
//!
//! ### UART Feed Mode, simulation from PC
//! ```powershell
//! cargo build --release --target xtensa-esp32s3-none-elf --features uart-feed
//! ```
//! Receives ADC values from PC via UART.
//! PC sends "2048\n" → ESP32-S3 responds "-1\n", "0\n", or "1\n".
//!
//! ## Pipeline
//!   ADC/UART → moving avg filter → ring buffer → inference → LED/Buzzer → UART log
//!
//! Target: xtensa-esp32s3-none-elf
//! DISCLAIMER: Educational prototype only. Not a medical device.

#![no_std]
#![no_main]

// ── Pure Rust modules ─────────────────────────────────────────────────────────
mod ring_buffer;
mod filter;
mod inference;
mod model_weights;
mod logger;

// ── UART Feed Mode module ─────────────────────────────────────────────────────
#[cfg(feature = "uart-feed")]
mod uart_feed;

// ── esp-hal imports ───────────────────────────────────────────────────────────
use esp_hal::{
    gpio::{Level, Output, OutputConfig},
};

#[cfg(not(feature = "uart-feed"))]
use esp_hal::delay::Delay;

#[cfg(not(feature = "uart-feed"))]
use esp_hal::analog::adc::{Adc, AdcConfig, Attenuation};

#[cfg(feature = "uart-feed")]
use esp_hal::uart::{Config as UartConfig, Uart};

use xtensa_lx_rt::entry;
use esp_backtrace as _;

// ESP-IDF app descriptor required by newer espflash.
esp_bootloader_esp_idf::esp_app_desc!();

// ── Constants ─────────────────────────────────────────────────────────────────
/// Sampling rate in Hz. 250 Hz = 4 ms period.
const SAMPLE_RATE_HZ: u32 = 250;

/// Sampling interval in microseconds.
const SAMPLE_INTERVAL_US: u32 = 1_000_000 / SAMPLE_RATE_HZ;

/// Ring buffer size in samples. 128 samples @ 250 Hz = 512 ms window.
const RING_BUF_SIZE: usize = 128;

/// Moving average filter window.
const FILTER_WINDOW: usize = 8;

// ── Entry point ───────────────────────────────────────────────────────────────
#[entry]
fn main() -> ! {
    // ── 1. Initialize ESP32-S3 peripherals ───────────────────────────────────
    let peripherals = esp_hal::init(esp_hal::Config::default());

    // ── 2. GPIO: LED and Buzzer ──────────────────────────────────────────────
    //
    // ESP32-S3 GPIO mapping:
    //   LED    → GPIO2
    //   Buzzer → GPIO21
    //
    // Note:
    //   GPIO25 exists on classic ESP32, but not on ESP32-S3.
    let mut led = Output::new(
        peripherals.GPIO2,
        Level::Low,
        OutputConfig::default(),
    );

    let mut buzzer = Output::new(
        peripherals.GPIO21,
        Level::Low,
        OutputConfig::default(),
    );

    // ── 3A. ADC Mode: ECG input from GPIO4 ───────────────────────────────────
    //
    // ESP32-S3 ADC1 supports GPIO1–GPIO10.
    // GPIO34 was used on classic ESP32, but it is not valid ADC input here.
    #[cfg(not(feature = "uart-feed"))]
    let (mut adc, mut ecg_pin) = {
        let mut adc_config = AdcConfig::new();

        let pin = adc_config.enable_pin(
            peripherals.GPIO4,
            Attenuation::_11dB,
        );

        let adc = Adc::new(
            peripherals.ADC1,
            adc_config,
        );

        (adc, pin)
    };

    // ── 3B. UART Feed Mode: PC sends ADC samples over UART0 ──────────────────
    //
    // Common ESP32-S3 UART0 pins:
    //   GPIO44 = UART0 RX
    //   GPIO43 = UART0 TX
    //
    // This replaces the old direct-register ESP32 UART0 code.
    //
    // Important:
    //   Do not use esp_println! in UART-feed mode because uart_reader owns UART0.
    #[cfg(feature = "uart-feed")]
    let mut uart_reader = {
        let uart_config = UartConfig::default().with_baudrate(115_200);

        let uart = Uart::new(
            peripherals.UART0,
            uart_config,
        )
        .unwrap()
        .with_rx(peripherals.GPIO44)
        .with_tx(peripherals.GPIO43);

        let mut reader = uart_feed::UartFeedReader::new(uart);

        reader.send_text("# UART_FEED_MODE ready. Send ADC values, e.g. 2048\\n\n");

        reader
    };

    // ── 4. Delay, only needed in real ADC mode ────────────────────────────────
    #[cfg(not(feature = "uart-feed"))]
    let delay = Delay::new();

    // ── 5. Signal processing state ───────────────────────────────────────────
    let mut ring_buf = ring_buffer::RingBuffer::<i32, RING_BUF_SIZE>::new();
    let mut filt = filter::MovingAverageState::<FILTER_WINDOW>::new();

    // ── 6. Application state variables ───────────────────────────────────────
    let mut time_ms: u32 = 0;
    let mut alert_active: bool = false;

    // ── 7. UART CSV header, only for ADC mode ────────────────────────────────
    #[cfg(not(feature = "uart-feed"))]
    logger::log_header();

    // ── 8. Main loop ─────────────────────────────────────────────────────────
    loop {
        // ── a) Get next sample ───────────────────────────────────────────────

        // ADC Mode: wait 4 ms, then read from GPIO4.
        #[cfg(not(feature = "uart-feed"))]
        let adc_raw: u16 = {
            delay.delay_micros(SAMPLE_INTERVAL_US);
            nb::block!(adc.read_oneshot(&mut ecg_pin)).unwrap_or(2048)
        };

        // UART Feed Mode: block until PC sends one ADC sample.
        #[cfg(feature = "uart-feed")]
        let adc_raw: u16 = uart_reader.read_sample().unwrap_or(2048);

        // ── b) Moving average filter ─────────────────────────────────────────
        let filtered: i32 = filt.push_and_average(adc_raw as i32);

        // ── c) Push into ring buffer ─────────────────────────────────────────
        ring_buf.push(filtered);

        // Default per-loop outputs.
        let mut prediction: u8 = 0;
        let mut inference_us: u32 = 0;
        let mut alert_latency_ms: u32 = 0;

        // ── d) Inference, runs once buffer is full ───────────────────────────
        if ring_buf.is_full() {
            let window: &[i32] = ring_buf.as_slice();

            prediction = inference::infer(window);

            // Placeholder timing.
            // Real timing can be measured with CPU cycle counter later.
            inference_us = 25;

            // ── e) Alert state machine ───────────────────────────────────────
            if prediction == 1 {
                if !alert_active {
                    alert_active = true;
                    alert_latency_ms = 0;
                }

                led.set_high();
                buzzer.set_high();
            } else {
                if alert_active {
                    alert_active = false;
                }

                led.set_low();
                buzzer.set_low();
                alert_latency_ms = 0;
            }
        }

        // ── f) Output: depends on build mode ─────────────────────────────────

        // ADC Mode: log full CSV line over UART for post-analysis.
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

        // UART Feed Mode: send only prediction back to PC.
        //
        // Protocol:
        //   "-1\n" = buffer not full yet
        //   "0\n"  = normal
        //   "1\n"  = abnormal
        #[cfg(feature = "uart-feed")]
        {
            let pred_out: i8 = if ring_buf.is_full() {
                prediction as i8
            } else {
                -1
            };

            uart_reader.send_prediction(pred_out);
        }

        // ── g) Advance timestamp ─────────────────────────────────────────────
        time_ms = time_ms.wrapping_add(SAMPLE_INTERVAL_US / 1000);
    }
}