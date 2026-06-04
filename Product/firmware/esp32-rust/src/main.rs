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
//!   ADC/UART → moving avg filter → ring buffer → inference → debounce → LED/Buzzer → UART log
//!
//! ## Timing
//!   Inference latency is measured using the Xtensa CPU cycle counter at 240 MHz.
//!   CSV logging is throttled to every LOG_EVERY_N samples to avoid disturbing
//!   the 250 Hz sample loop with slow UART writes.
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

/// CPU clock frequency in Hz (ESP32-S3 default: 240 MHz).
/// Used to convert CPU cycle counts to microseconds.
const CPU_FREQ_HZ: u32 = 240_000_000;

// ── Debounce Constants ────────────────────────────────────────────────────────
/// Number of consecutive ABNORMAL windows required before triggering alert.
/// Prevents false positives from single noisy predictions.
const DEBOUNCE_ABNORMAL: u8 = 3;

/// Number of consecutive NORMAL windows required before clearing alert.
/// Prevents alert flickering on borderline signals.
const DEBOUNCE_NORMAL: u8 = 5;

// ── Logging Rate ──────────────────────────────────────────────────────────────
/// In ADC mode, log a CSV row only every N samples.
/// Reduces UART output from 250 writes/sec to 250/N writes/sec.
/// At 115200 baud each CSV row ≈ 0.7 ms; logging every sample ≈ 17% overhead.
/// Logging every 8th sample ≈ 2% overhead.
#[cfg(not(feature = "uart-feed"))]
const LOG_EVERY_N: u32 = 8;

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

    // ── 5. Signal processing state ─────────────────────────────────────────
    let mut ring_buf = ring_buffer::RingBuffer::<i32, RING_BUF_SIZE>::new();
    let mut filt = filter::MovingAverageState::<FILTER_WINDOW>::new();

    // ── 6. Application state variables ───────────────────────────────────
    let mut time_ms: u32 = 0;
    let mut alert_active: bool = false;

    // ── Debounce counters ────────────────────────────────────────────────
    //
    // abnormal_streak: consecutive windows predicting abnormal
    // normal_streak  : consecutive windows predicting normal
    //
    // Alert fires when abnormal_streak reaches DEBOUNCE_ABNORMAL.
    // Alert clears when normal_streak  reaches DEBOUNCE_NORMAL.
    let mut abnormal_streak: u8 = 0;
    let mut normal_streak: u8 = 0;

    // ── Alert latency tracking ────────────────────────────────────────────
    //
    // Records time_ms at which the first abnormal prediction occurred in
    // the current run. Used to compute alert_latency_ms.
    let mut first_abnormal_ms: u32 = 0;

    // ── Sample counter for UART log throttling (ADC mode only) ────────────
    #[cfg(not(feature = "uart-feed"))]
    let mut log_counter: u32 = 0;

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

        // ── d) Inference, runs once buffer is full ───────────────────────
        if ring_buf.is_full() {
            let window: &[i32] = ring_buf.as_slice();

            // ── Real inference timing via CPU cycle counter ───────────────
            //
            // xtensa_lx::timer::get_cycle_count() reads the Xtensa CCOUNT
            // register (increments every CPU cycle at 240 MHz).
            // Δcycles / CPU_FREQ_HZ × 1_000_000 = microseconds.
            let cycles_before = xtensa_lx::timer::get_cycle_count();
            prediction = inference::infer(window);
            let cycles_after = xtensa_lx::timer::get_cycle_count();

            let delta_cycles = cycles_after.wrapping_sub(cycles_before);
            // Avoid division by zero; saturate at u32::MAX µs if overflow.
            inference_us = (delta_cycles as u64 * 1_000_000 / CPU_FREQ_HZ as u64)
                .min(u32::MAX as u64) as u32;

            // ── e) Debounce state machine ─────────────────────────────────
            //
            // Require DEBOUNCE_ABNORMAL consecutive abnormal predictions
            // before activating alert. Require DEBOUNCE_NORMAL consecutive
            // normal predictions before clearing alert.
            // This eliminates single-window false positives and prevents
            // buzzer/LED flickering on borderline signals.
            if prediction == 1 {
                abnormal_streak = abnormal_streak.saturating_add(1);
                normal_streak = 0;

                if abnormal_streak >= DEBOUNCE_ABNORMAL {
                    if !alert_active {
                        alert_active = true;
                        first_abnormal_ms = time_ms;
                    }
                }
            } else {
                normal_streak = normal_streak.saturating_add(1);
                abnormal_streak = 0;

                if normal_streak >= DEBOUNCE_NORMAL {
                    if alert_active {
                        alert_active = false;
                    }
                }
            }

            // ── f) Alert latency ─────────────────────────────────────────
            //
            // Alert latency = time from first abnormal prediction to when
            // the alert actually fired (after debounce confirmation).
            alert_latency_ms = if alert_active {
                time_ms.wrapping_sub(first_abnormal_ms)
            } else {
                0
            };

            // ── g) Drive LED and buzzer ────────────────────────────────────
            if alert_active {
                led.set_high();
                buzzer.set_high();
            } else {
                led.set_low();
                buzzer.set_low();
            }
        }

        // ── h) Output: depends on build mode ─────────────────────────────

        // ADC Mode: log CSV line over UART for post-analysis.
        // Throttled to every LOG_EVERY_N samples to reduce UART overhead.
        // At 250 Hz and LOG_EVERY_N=8: logs at ~31 Hz, UART overhead < 2%.
        #[cfg(not(feature = "uart-feed"))]
        {
            log_counter = log_counter.wrapping_add(1);
            if log_counter >= LOG_EVERY_N {
                log_counter = 0;
                logger::log_csv(
                    time_ms,
                    adc_raw as i32,
                    filtered,
                    inference_us,
                    prediction,
                    alert_active as u8,
                    alert_latency_ms,
                );
            }
        }

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