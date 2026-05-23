# RT-QTinyECG ESP32 Rust Firmware

This directory contains the `no_std` embedded Rust firmware for the ECG detection prototype.

---

## Prerequisites

### 1. Install `espup` and the ESP32 Rust Toolchain

```bash
cargo install espup
espup install
```

On Linux/macOS, source the environment:
```bash
source ~/export-esp.sh
```

On Windows (PowerShell):
```powershell
. $HOME\export-esp.ps1
```

### 2. Install Flash Tool

```bash
cargo install espflash
```

---

## Build

```bash
cd firmware/esp32-rust
cargo build --release
```

Expected output: `target/xtensa-esp32-none-elf/release/ecg-esp32`

---

## Flash

Connect ESP32 via USB and run:

```bash
espflash flash target/xtensa-esp32-none-elf/release/ecg-esp32 --monitor
```

Or using cargo-espflash:
```bash
cargo espflash flash --release --monitor
```

---

## HAL Version Notes

This firmware targets **esp-hal 0.18.x**. If you use a different version:

### ADC API Changes

| Version | ADC Import | ADC Read |
|---|---|---|
| < 0.15 | `use esp_hal::adc::{Adc, AdcConfig, ...}` | `.read()` returns `Result` |
| 0.15–0.17 | `use esp_hal::analog::adc::{...}` | `nb::block!(.read_oneshot(...))` |
| 0.18+ | `use esp_hal::analog::adc::{...}` | `nb::block!(.read_oneshot(...))` |

### GPIO API Changes

| Version | Output API |
|---|---|
| < 0.15 | `gpio34.into_push_pull_output(); pin.set_high().unwrap()` |
| 0.18+ | `Output::new(gpio2, Level::Low); pin.set_high()` |

### Delay API Changes

| Version | Delay |
|---|---|
| < 0.15 | `Delay::new(&clocks); delay.delay_us(N)` |
| 0.18+ | `Delay::new(&clocks); delay.delay_micros(N)` |

---

## Adapting to Different esp-hal Versions

If you get compile errors, look for `// HAL-VERSION-NOTE` comments throughout the source files.
These mark the exact lines that need version-specific adaptation.

Most common adjustments:
1. **ADC initialization** in `src/adc_reader.rs`
2. **GPIO Output creation** in `src/main.rs` and `src/alert.rs`
3. **Delay call** in `src/main.rs`

---

## UART Serial Output

The firmware logs CSV over UART0 at **115200 baud**.

Connect with any terminal:
```bash
# Linux
screen /dev/ttyUSB0 115200

# macOS
screen /dev/cu.usbserial-XXXX 115200

# Windows: PuTTY, Tera Term, or Arduino Serial Monitor at 115200 baud
```

Capture to file:
```bash
cat /dev/ttyUSB0 > data/hardware_log.csv  # Linux
```

---

## Simulated Signal (No Hardware)

If you don't have an AD8232, you can modify `src/adc_reader.rs` to return a
simulated ECG value instead of reading the ADC:

```rust
pub fn read_ecg_adc(&mut self) -> u16 {
    // Simulate a sine wave for testing (no hardware needed)
    static mut COUNTER: u32 = 0;
    unsafe {
        COUNTER = COUNTER.wrapping_add(1);
        // Simple sawtooth for testing
        let val = (COUNTER % 4096) as u16;
        val
    }
}
```

---

## Memory Usage

| Region | Usage | ESP32 Total |
|---|---|---|
| Flash (firmware) | ~50–150 KB (estimated) | 4 MB |
| IRAM (code) | ~10–30 KB | 128 KB |
| DRAM (data) | ~2–10 KB | 520 KB |
| Ring buffer | 128 × 4 = 512 bytes | – |
| Model weights | ~100 bytes | – |

ESP32 has plenty of RAM for this application.

---

## Disclaimer

This firmware is for **educational use only**.
NOT a medical device. NOT for clinical diagnosis or treatment.
