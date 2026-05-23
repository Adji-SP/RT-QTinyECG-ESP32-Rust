//! alert.rs
//! =========
//! LED and buzzer control abstraction for ECG alert system.
//!
//! The alert system activates when the inference engine detects an
//! abnormal ECG pattern. It controls two output devices:
//!   - LED on GPIO2 (built-in LED on most ESP32 DevKit V1 boards)
//!   - Buzzer on GPIO25 (active buzzer, HIGH = sound ON)
//!
//! # Alert Logic:
//!   - alert_on()  : Turn LED and buzzer ON (abnormal detected)
//!   - alert_off() : Turn LED and buzzer OFF (normal signal)
//!   - set_alert() : Set state from a boolean
//!
//! # Hardware Notes:
//!   - GPIO2 LED: HIGH = LED ON (positive logic on DevKit V1)
//!     Some boards have inverse logic (LOW = ON). Check your board.
//!   - GPIO25 (Buzzer): HIGH = active buzzer sounds.
//!     GPIO25 shares with DAC2 on ESP32. If you need DAC2, move buzzer
//!     to another GPIO (e.g., GPIO26).
//!   - Both are configured as push-pull output, driven from 3.3V GPIO.
//!     For high-current buzzer, use a transistor (NPN + 1kΩ base resistor).
//!
//! # HAL-VERSION-NOTE:
//!   Output GPIO API in esp-hal 0.18:
//!     use esp_hal::gpio::{Level, Output};
//!     let mut led = Output::new(io.pins.gpio2, Level::Low);
//!     led.set_high();
//!   In older versions (<0.15):
//!     let mut led = io.pins.gpio2.into_push_pull_output();
//!     led.set_high().unwrap();
//!
//! # DISCLAIMER: Educational prototype only. Not for clinical use.

// Import GPIO output types from esp-hal
// HAL-VERSION-NOTE: These import paths are for esp-hal 0.18.x.
use esp_hal::gpio::{GpioPin, Level, Output};

/// Alert controller managing LED and buzzer GPIO outputs.
///
/// Holds ownership of both output pins and tracks the current alert state.
///
/// # HAL-VERSION-NOTE:
/// The generic types on Output may differ between versions.
/// In 0.18: `Output<'static, GpioPin<N>>`
/// In newer: `Output<GpioPin<N>>` (no lifetime)
pub struct AlertController {
    /// LED output (GPIO2, built-in LED)
    led:    Output<'static, GpioPin<2>>,

    /// Buzzer output (GPIO25)
    buzzer: Output<'static, GpioPin<25>>,

    /// Current alert state (true = alert is active)
    active: bool,
}

impl AlertController {
    /// Create a new alert controller.
    ///
    /// # Arguments
    /// * `led_pin`    - GPIO2 output, already configured as Output::new(...)
    /// * `buzzer_pin` - GPIO25 output, already configured as Output::new(...)
    ///
    /// Both pins start in the LOW (off) state.
    pub fn new(
        led_pin:    Output<'static, GpioPin<2>>,
        buzzer_pin: Output<'static, GpioPin<25>>,
    ) -> Self {
        Self {
            led:    led_pin,
            buzzer: buzzer_pin,
            active: false,
        }
    }

    /// Activate the alert: turn LED and buzzer ON.
    ///
    /// This is called when inference returns prediction = 1 (Abnormal).
    ///
    /// # HAL-VERSION-NOTE:
    /// In 0.18+: `self.led.set_high()` (no Result, infallible)
    /// In older:  `self.led.set_high().unwrap()` (returns Result)
    #[inline]
    pub fn alert_on(&mut self) {
        if !self.active {
            // HAL-VERSION-NOTE: set_high() is infallible in 0.18+
            self.led.set_high();
            self.buzzer.set_high();
            self.active = true;
        }
        // If already active, no re-write needed (GPIO state doesn't change)
    }

    /// Deactivate the alert: turn LED and buzzer OFF.
    ///
    /// This is called when inference returns prediction = 0 (Normal).
    #[inline]
    pub fn alert_off(&mut self) {
        if self.active {
            self.led.set_low();
            self.buzzer.set_low();
            self.active = false;
        }
    }

    /// Set alert state from a boolean.
    ///
    /// `true`  → alert_on()
    /// `false` → alert_off()
    ///
    /// This is a convenience wrapper to avoid if/else at the call site.
    #[inline]
    pub fn set_alert(&mut self, active: bool) {
        if active {
            self.alert_on();
        } else {
            self.alert_off();
        }
    }

    /// Return the current alert state.
    ///
    /// `true`  = Alert is ON (abnormal detected)
    /// `false` = Alert is OFF (normal signal)
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Toggle the alert state (for testing or blinking).
    ///
    /// Not used in normal operation. Useful for self-test at startup.
    pub fn toggle(&mut self) {
        if self.active {
            self.alert_off();
        } else {
            self.alert_on();
        }
    }

    /// Blink the LED N times (for startup indication).
    ///
    /// This is a blocking function that blinks the LED `n` times,
    /// using a simple cycle-based delay approximation.
    /// Use sparingly: blocks the sampling loop for `n × 2 × delay_cycles`.
    ///
    /// Only use this during initialization, not in the main loop.
    ///
    /// # Arguments
    /// * `n`             - Number of blinks
    /// * `delay_cycles`  - Approximate delay cycles per half-blink
    ///                     (240 cycles ≈ 1 µs at 240 MHz)
    pub fn startup_blink(&mut self, n: u8, delay_cycles: u32) {
        for _ in 0..n {
            self.led.set_high();
            // Simple busy-wait delay
            for _ in 0..delay_cycles {
                unsafe { core::arch::asm!("nop") };
            }
            self.led.set_low();
            for _ in 0..delay_cycles {
                unsafe { core::arch::asm!("nop") };
            }
        }
    }
}
