//! adc_reader.rs
//! ==============
//! ADC abstraction for reading the ECG analog signal from GPIO34.
//!
//! The ESP32 has two ADC units:
//!   - ADC1: GPIO32–GPIO39 (can be used even when Wi-Fi is active)
//!   - ADC2: GPIO0–GPIO15,25–27 (conflicts with Wi-Fi; prefer ADC1)
//!
//! We use ADC1, Channel 6 (GPIO34) for ECG input.
//! GPIO34 is an input-only pin on ESP32 (no pullup/pulldown).
//!
//! # ESP32 ADC Characteristics (important for calibration):
//!   - 12-bit resolution: 0–4095 (corresponding to 0–3.3V input)
//!   - Nonlinear response near 0V and 3.3V (usable range: ~0.1V–3.0V)
//!   - Built-in attenuation settings: 0dB, 2.5dB, 6dB, 11dB
//!     - 11dB attenuation: 0–3.3V range (widest, most nonlinear near edges)
//!     - 6dB  attenuation: 0–2.2V range (better linearity, narrower)
//!   - For ECG: AD8232 output is centered at VCC/2 = 1.65V
//!     → Use 11dB or 6dB attenuation depending on signal swing
//!
//! # HAL-VERSION-NOTES:
//!   The esp-hal ADC API has changed significantly between versions.
//!   This implementation targets esp-hal 0.18.x.
//!   For other versions, look for `// HAL-VERSION-NOTE` comments below.
//!   See README_FIRMWARE.md for migration guide.
//!
//! # Lead-off Detection:
//!   The AD8232 has two lead-off detection outputs:
//!     LO+ → GPIO32 (HIGH when lead is off)
//!     LO- → GPIO33 (HIGH when lead is off)
//!   When either lead is off, the ADC input floats and gives invalid data.
//!   We detect this by checking if GPIO32 or GPIO33 is HIGH.
//!   If leads are off, return 0 from read_ecg_adc() to indicate invalid data.
//!
//! # DISCLAIMER: Educational prototype only. Not for clinical use.

// HAL-VERSION-NOTE: ADC imports for esp-hal 0.18.x
// In 0.18, ADC is accessed via:
//   use esp_hal::analog::adc::{Adc, AdcConfig, AdcPin, Attenuation};
// In older versions (<0.15):
//   use esp_hal::adc::{Adc, AdcConfig, AdcPin, Attenuation};
// In very new versions (>0.18), check the changelog.
use esp_hal::analog::adc::{Adc, AdcConfig, AdcPin, Attenuation};
use esp_hal::peripherals::ADC1;

// HAL-VERSION-NOTE: GPIO input for lead-off detection
// In 0.18: use esp_hal::gpio::{Input, Pull};
use esp_hal::gpio::{GpioPin, Input, Pull};

/// ECG ADC reader abstraction.
///
/// Wraps the ESP32 ADC1 peripheral and provides a simple interface
/// to read one ECG sample.
///
/// # HAL-VERSION-NOTE:
/// The generic types here (e.g., `AdcPin<GpioPin<34>, ADC1, ...>`) may
/// change between esp-hal versions. If you get compilation errors:
///   1. Check the esp-hal version in Cargo.toml
///   2. Refer to esp-hal examples for the current ADC initialization pattern
///   3. See README_FIRMWARE.md for version-specific notes
pub struct AdcReader {
    /// ADC driver instance
    // HAL-VERSION-NOTE: The Adc struct generic parameters differ by version.
    // In 0.18: Adc<'static, ADC1>
    // In some versions: no lifetime parameter.
    adc: Adc<'static, ADC1>,

    /// ADC pin configuration for GPIO34 (ADC1_CH6)
    // HAL-VERSION-NOTE: AdcPin type may differ.
    // In 0.18: AdcPin<GpioPin<34>, ADC1, AdcCalNoCalNoEfuse>
    // In older: AdcPin<Gpio34, ADC1, AdcNoCalibration>
    // The third generic is the calibration type; use the default for your version.
    ecg_pin: AdcPin<GpioPin<34>, ADC1>,

    // ── Optional: Lead-off detection pins ─────────────────────────────────────
    // Uncomment these if you have the AD8232 LO+ and LO- connected.
    // HAL-VERSION-NOTE: Input<'static, GpioPin<N>> in 0.18.
    // lo_plus:  Input<'static, GpioPin<32>>,
    // lo_minus: Input<'static, GpioPin<33>>,
}

impl AdcReader {
    /// Initialize the ADC reader for ECG sampling.
    ///
    /// # Arguments
    /// * `adc1_peripheral` - The ADC1 peripheral from `Peripherals::take()`
    ///
    /// # HAL-VERSION-NOTE:
    /// This function signature and implementation may need adjustment.
    /// In esp-hal 0.18, ADC initialization looks like:
    ///
    /// ```rust,no_run
    /// let mut adc_config = AdcConfig::new();
    /// let ecg_pin = adc_config.enable_pin(
    ///     io.pins.gpio34,
    ///     Attenuation::Attenuation11dB,
    /// );
    /// let adc = Adc::new(peripherals.ADC1, adc_config);
    /// ```
    ///
    /// In older versions (<0.15):
    /// ```rust,no_run
    /// let mut adc_config = AdcConfig::new();
    /// let ecg_pin = adc_config.enable_pin(
    ///     io.pins.gpio34.into_analog(),
    ///     Attenuation::Attenuation11dB,
    /// );
    /// let adc = Adc::<ADC1>::new(peripherals.ADC1, adc_config);
    /// ```
    ///
    /// If you cannot get the exact API to work, see the fallback approach
    /// in README_FIRMWARE.md (using esp-idf-hal or direct register access).
    pub fn new(
        adc1: ADC1,
        // gpio34: GpioPin<34>,  // Uncomment if your version requires the pin
    ) -> Self {
        // ── ADC Configuration ─────────────────────────────────────────────────
        let mut adc_config = AdcConfig::new();

        // Configure GPIO34 as ADC input with 11 dB attenuation
        // 11 dB attenuation → input range 0–3.3V (ESP32 specific)
        // For AD8232 with VCC=3.3V, output is centered at ~1.65V
        // This setting gives the widest input range.
        //
        // HAL-VERSION-NOTE: In some versions, you must call gpio34.into_analog()
        // before passing to enable_pin(). Try both if you get type errors.
        let ecg_pin = adc_config.enable_pin(
            // gpio34,              // Use this if your version requires the pin argument
            unsafe { GpioPin::<34>::steal() }, // For demo: steal pin. In real code, pass via arg.
            Attenuation::Attenuation11dB,
        );

        // Initialize ADC1 with the configuration
        let adc = Adc::new(adc1, adc_config);

        Self { adc, ecg_pin }
    }

    /// Read one ECG ADC sample from GPIO34.
    ///
    /// # Returns
    /// - 0–4095: Valid ADC sample (12-bit, 0 = 0V, 4095 = 3.3V)
    /// - 0: Leads off (AD8232 LO+/LO- detected, if connected)
    ///
    /// # Real-time behavior:
    /// The ESP32 ADC conversion takes approximately 20–100 µs depending
    /// on clock and configuration. This is well within the 4 ms sampling budget.
    ///
    /// # HAL-VERSION-NOTE:
    /// The read method differs by version:
    ///   - 0.18+: `nb::block!(self.adc.read_oneshot(&mut self.ecg_pin)).unwrap_or(2048)`
    ///   - Older: `self.adc.read(&mut self.ecg_pin).unwrap_or(2048)`
    ///
    /// If the ADC read fails (Err), we return 2048 (midpoint) as a safe fallback.
    pub fn read_ecg_adc(&mut self) -> u16 {
        // ── Optional: Check lead-off pins ─────────────────────────────────────
        // Uncomment if LO+/LO- pins are connected to GPIO32/GPIO33:
        //
        // if self.lo_plus.is_high() || self.lo_minus.is_high() {
        //     // Leads are off: return 0 to signal invalid data
        //     return 0;
        // }

        // ── Read ADC ──────────────────────────────────────────────────────────
        // HAL-VERSION-NOTE: nb::block!() is the non-blocking to blocking adapter.
        // In 0.18, ADC reads use the nb crate's Result/WouldBlock pattern.
        // In older versions, .read() was synchronous.
        let result = nb::block!(self.adc.read_oneshot(&mut self.ecg_pin));

        match result {
            Ok(sample) => sample,
            Err(_) => {
                // ADC read error: return midpoint as safe fallback
                // This prevents downstream code from seeing an extreme value.
                // In production, log this error and consider re-initializing ADC.
                2048
            }
        }
    }

    /// Read multiple ADC samples and return the average.
    ///
    /// Oversampling (reading multiple samples per interval) reduces ADC noise.
    /// The cost is proportional to `count` × single_read_time.
    /// Use count=4 for 4× oversampling (costs ~80–400 µs extra).
    ///
    /// At 250 Hz with 4 ms budget, count=2–4 is usually feasible.
    ///
    /// # Arguments
    /// * `count` - Number of samples to average (1 = no oversampling)
    pub fn read_oversampled(&mut self, count: u8) -> u16 {
        let n = count.max(1) as u32;
        let mut sum: u32 = 0;
        for _ in 0..n {
            sum += self.read_ecg_adc() as u32;
        }
        (sum / n) as u16
    }
}
