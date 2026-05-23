//! alert.rs — Placeholder for esp-hal 1.x
//!
//! In esp-hal 1.x, GPIO Output lifetime handling makes wrapping pins in a
//! struct tricky. Alert logic is inlined directly in main.rs instead:
//!
//!   let mut led    = Output::new(peripherals.GPIO2,  Level::Low);
//!   let mut buzzer = Output::new(peripherals.GPIO25, Level::Low);
//!   // In loop:
//!   if prediction == 1 { led.set_high(); buzzer.set_high(); }
//!   else               { led.set_low();  buzzer.set_low();  }
//!
//! This file is kept for reference. It is not compiled (no mod declaration in main.rs).
