//! adc_reader.rs — Placeholder for esp-hal 1.x
//!
//! In esp-hal 1.x, ADC initialization is inlined directly in main.rs
//! to avoid struct lifetime/type-naming complexity.
//!
//! The ADC logic is now in main.rs:
//!   let mut adc_config = AdcConfig::new();
//!   let mut ecg_pin = adc_config.enable_pin(peripherals.GPIO34, Attenuation::_11dB);
//!   let mut adc = Adc::new(peripherals.ADC1, adc_config);
//!   let sample: u16 = nb::block!(adc.read_oneshot(&mut ecg_pin)).unwrap_or(2048);
//!
//! This file is kept for reference. It is not compiled (no mod declaration in main.rs).
//!
//! For esp-hal 0.18.x (older toolchain), see the version in git history.
