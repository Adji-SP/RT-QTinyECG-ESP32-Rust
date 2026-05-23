# RT-QTinyECG-ESP32-Rust

**Real-Time Quantized ECG Detection on ESP32 Using Embedded Rust**

> ⚠️ **DISCLAIMER**: This project is for **educational and embedded systems learning purposes only**.
> It is **NOT** a clinical medical device, NOT validated for diagnostic use, and must **NOT** be used
> for real medical decisions. If you have cardiac concerns, consult a licensed medical professional.

---

## Overview

RT-QTinyECG-ESP32-Rust is an educational embedded systems prototype that demonstrates:

- Real-time ECG-like signal acquisition on ESP32 using Rust (`no_std`)
- Fixed-rate ADC sampling via timer ISR at **250 Hz**
- Circular ring buffer for signal windowing
- Moving-average digital filtering
- Lightweight **quantized inference** (threshold + tiny MLP with int8 weights)
- Normal / Abnormal classification decision
- LED and Buzzer safety alert
- CSV serial logging over UART
- PC-side Python simulation and GNUPlot visualization
- Real-time measurability: sampling stability, inference time, alert latency

The system uses **Approach A** (ESP32 Rust `no_std` firmware) and **Approach B** (Python PC simulator) side by side.

---

## Features

| Feature | Status |
|---|---|
| 250 Hz fixed-rate ADC sampling | ✅ |
| Circular ring buffer (const generics) | ✅ |
| Moving average filter | ✅ |
| Baseline removal | ✅ |
| Threshold-based classifier | ✅ |
| Quantized tiny MLP (int8 weights) | ✅ |
| LED alert on GPIO2 | ✅ |
| Buzzer alert on GPIO25 | ✅ |
| Lead-off detection (GPIO32/33) | ✅ |
| CSV UART logging | ✅ |
| Python ECG simulator | ✅ |
| Python model training + quantization | ✅ |
| GNUPlot visualization | ✅ |
| Real-time metrics (latency, inference time) | ✅ |

---

## Repository Structure

```
RT-QTinyECG-ESP32-Rust/
├── README.md                        ← This file
├── LICENSE                          ← MIT License
├── requirements.txt                 ← Python dependencies
├── data/
│   ├── sample_ecg.csv               ← Generated dummy ECG data
│   ├── simulated_realtime_log.csv   ← Simulator output log
│   └── README_DATA.md               ← Data format description
├── python/
│   ├── generate_dummy_ecg.py        ← Generate dummy ECG CSV
│   ├── realtime_ecg_simulator.py    ← Simulate ESP32 pipeline on PC
│   ├── preprocessing.py             ← Signal preprocessing helpers
│   ├── train_simple_model.py        ← Train logistic/MLP classifier
│   ├── quantize_weights.py          ← Quantize float32 → int8
│   ├── export_rust_weights.py       ← Export weights to Rust source
│   └── metrics.py                   ← Evaluation metrics
├── firmware/
│   └── esp32-rust/
│       ├── README_FIRMWARE.md       ← Firmware build guide
│       ├── Cargo.toml               ← Rust manifest
│       ├── rust-toolchain.toml      ← Rust toolchain pin
│       ├── .cargo/
│       │   └── config.toml          ← Cargo build target config
│       └── src/
│           ├── main.rs              ← Firmware entry point
│           ├── ring_buffer.rs       ← Circular buffer
│           ├── filter.rs            ← Signal filters
│           ├── inference.rs         ← Inference engine
│           ├── model_weights.rs     ← Quantized int8 weights
│           ├── adc_reader.rs        ← ADC abstraction
│           ├── alert.rs             ← LED/Buzzer control
│           └── logger.rs            ← CSV UART logger
├── gnuplot/
│   ├── plot_ecg_signal.gp           ← Raw + filtered signal plot
│   ├── plot_alert_latency.gp        ← Alert latency plot
│   └── plot_inference_time.gp       ← Inference time plot
├── docs/
│   ├── block_diagram.md             ← System block diagram (Mermaid)
│   ├── flowchart.md                 ← Firmware flowchart (Mermaid)
│   ├── wiring_esp32_ad8232.md       ← Hardware wiring guide
│   ├── algorithm.md                 ← Algorithm explanation
│   ├── evaluation_metrics.md        ← Metrics guide
│   └── real_time_design.md          ← Real-time design rationale
└── images/
    └── placeholder.txt
```

---

## Hardware Requirements

| Component | Details |
|---|---|
| ESP32 DevKit V1 | Any generic ESP32 board |
| AD8232 ECG Module | Optional for real hardware |
| LED | Connected to GPIO2 (built-in LED) |
| Buzzer | Active buzzer on GPIO25 |
| USB Cable | For flashing and UART serial |
| Jumper Wires | For AD8232 connections |
| Breadboard | For prototyping |

### ESP32 Pin Mapping

| Signal | ESP32 GPIO | Notes |
|---|---|---|
| ECG Analog Input | GPIO34 (ADC1_CH6) | Input only, max 3.3V |
| Lead-Off LO+ | GPIO32 | AD8232 lead-off detect |
| Lead-Off LO- | GPIO33 | AD8232 lead-off detect |
| LED Alert | GPIO2 | Built-in LED, HIGH = ON |
| Buzzer Alert | GPIO25 | Active buzzer, HIGH = ON |
| GND | GND | Common ground |
| 3.3V | 3V3 | Power for AD8232 |

---

## Software Requirements

- Rust (ESP32 toolchain via `espup`)
- Python 3.9+
- GNUPlot 5.x
- `cargo` + `espflash`

---

## Installation Guide

### 1. Install Rust ESP32 Toolchain

```bash
# Install espup (ESP32 Rust toolchain manager)
cargo install espup
espup install

# Source environment (Linux/macOS)
source ~/export-esp.sh

# On Windows, run: export-esp.ps1

# Install espflash for flashing
cargo install espflash

# Install cargo-espflash (optional, for convenience)
cargo install cargo-espflash
```

### 2. Install Python Dependencies

```bash
cd RT-QTinyECG-ESP32-Rust
pip install -r requirements.txt
```

### 3. Install GNUPlot

- **Ubuntu/Debian**: `sudo apt install gnuplot`
- **macOS**: `brew install gnuplot`
- **Windows**: Download from http://www.gnuplot.info/

---

## Simulation First-Run (No Hardware Required)

Run these commands in order from the project root:

```bash
# Step 1: Generate dummy ECG data
python python/generate_dummy_ecg.py

# Step 2: Run the real-time simulator
python python/realtime_ecg_simulator.py

# Step 3: Calculate evaluation metrics
python python/metrics.py

# Step 4: Plot results with GNUPlot
gnuplot gnuplot/plot_ecg_signal.gp
gnuplot gnuplot/plot_alert_latency.gp
gnuplot gnuplot/plot_inference_time.gp
```

---

## Model Training and Weight Export

```bash
# Train a simple logistic/MLP classifier
python python/train_simple_model.py

# Quantize float32 weights to int8
python python/quantize_weights.py

# Export quantized weights to Rust source file
python python/export_rust_weights.py
```

After running `export_rust_weights.py`, the file
`firmware/esp32-rust/src/model_weights.rs` will be updated with real quantized weights.

---

## Build ESP32 Rust Firmware

```bash
cd firmware/esp32-rust

# Check the toolchain (must match rust-toolchain.toml)
rustup show

# Build (no flash)
cargo build --release

# Build and flash to connected ESP32
cargo espflash flash --release --monitor

# Or use espflash directly
espflash flash target/xtensa-esp32-none-elf/release/ecg-esp32 --monitor
```

See `firmware/esp32-rust/README_FIRMWARE.md` for detailed firmware setup instructions
and notes on HAL version-specific API differences.

---

## How to Flash ESP32

1. Connect ESP32 to PC via USB.
2. Identify the serial port:
   - Linux: `/dev/ttyUSB0` or `/dev/ttyACM0`
   - macOS: `/dev/cu.usbserial-XXXX`
   - Windows: `COM3` (check Device Manager)
3. Run:

```bash
espflash flash target/xtensa-esp32-none-elf/release/ecg-esp32 --port /dev/ttyUSB0 --monitor
```

---

## Reading Serial Output

The firmware outputs CSV lines over UART at 115200 baud:

```
time_ms,adc_value,filtered_value,inference_us,prediction,alert,alert_latency_ms
```

Example:
```
1000,2048,2051,45,0,0,0
1004,2060,2053,44,0,0,0
1008,3800,3100,46,1,1,2
```

Use any serial terminal: `minicom`, `screen`, PuTTY, or Arduino Serial Monitor at 115200 baud.

---

## How to Plot with GNUPlot

After running the simulator, open a terminal in the project root and run:

```bash
gnuplot gnuplot/plot_ecg_signal.gp
gnuplot gnuplot/plot_alert_latency.gp
gnuplot gnuplot/plot_inference_time.gp
```

PNG output images will be saved in the `images/` directory.

---

## Limitations

- **Not a medical device.** Educational prototype only.
- The tiny MLP classifier uses placeholder/simplified weights. Real deployment needs proper training data.
- The Rust firmware may need HAL version-specific adjustments (see `README_FIRMWARE.md`).
- ADC on ESP32 has nonlinearity; use an external ADC for higher precision if needed.
- Inference is window-based; the system adds a small latency of one buffer window (128 samples @ 250 Hz = ~512 ms).
- The buzzer on GPIO25 may conflict with DAC2; use GPIO26 or another GPIO if needed.

---

## Future Work

- Integrate TensorFlow Lite Micro in Rust via FFI for more complex models.
- Add Wi-Fi/MQTT logging to a cloud dashboard.
- Use an external ADC (ADS1115) for higher ECG signal quality.
- Implement Pan-Tompkins R-peak detection for real HRV analysis.
- Add BLE streaming to a mobile app.
- Use RTOS task scheduling (`esp-idf-hal`) for better real-time guarantees.
- Add an OLED display for local visualization.
- Power profiling with battery life estimation.

---

## License

MIT License. See `LICENSE` for details.

---

## Contributing

This is an educational project. Contributions, improvements, and corrections are welcome.
Please open an issue or pull request on GitHub.

---

*RT-QTinyECG-ESP32-Rust — Educational Embedded Systems Prototype*
*Not for clinical use. Made for learning.*
