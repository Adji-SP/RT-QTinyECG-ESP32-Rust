# RT-QTinyECG-ESP32-Rust

Real-time quantized ECG-like signal classification on ESP32-S3 using Embedded Rust, with a Python training and evaluation pipeline.

> Safety notice: This repository is for education and embedded systems learning only. It is not a clinical medical device, has not been medically validated, and must not be used for diagnosis, monitoring, or treatment decisions.

## Overview

The project demonstrates a small end-to-end TinyML workflow:

1. Generate a labeled synthetic ECG-like signal.
2. Train a tiny sklearn MLP on firmware-compatible features.
3. Quantize the model weights to int8.
4. Export the quantized weights into Rust source.
5. Run the same pipeline in ESP32-S3 firmware.
6. Evaluate PC float32 predictions against ESP32/int8 predictions.
7. Generate comparison plots and optimization reports.

The current embedded model is a 5 -> 8 -> 1 MLP using int8 weights and i32 accumulators. The firmware extracts five features from a 128-sample sliding window and normalizes them using the same integer-style preprocessing used during training.

## Current Results

Latest dry-run evaluation after pipeline alignment:

| Metric | PC float32 | ESP32/int8 dry-run |
|---|---:|---:|
| Accuracy | 96.5% | 95.4% |
| Precision | 1.000 | 1.000 |
| Recall | 0.917 | 0.892 |
| F1-score | 0.957 | 0.943 |
| Quantization delta | - | 1.05% |
| PC/int8 agreement | - | 98.9% |

The dry-run path simulates the quantized ESP32 inference from `data/quantized_weights.npz`. A real hardware run still requires flashing the ESP32-S3.

## Repository Layout

```text
Product/
  README.md
  requirements.txt
  run_simulation.bat       PC-only full simulation pipeline
  run_uart_eval.bat        ESP32-S3 UART-feed evaluation pipeline
  run_all_plots.bat        Regenerate GNUPlot charts
  data/                    Generated CSVs, model files, reports, plot data
  docs/                    Project documentation
  firmware/esp32-rust/     no_std ESP32-S3 Rust firmware
  gnuplot/                 Plot scripts
  images/                  Generated PNG charts
  python/                  Data, training, quantization, evaluation scripts
```

Important generated files:

| File | Purpose |
|---|---|
| `data/sample_ecg.csv` | Synthetic ECG-like samples and labels |
| `data/model.pkl` | Trained sklearn MLP |
| `data/scaler.pkl` | Kept for compatibility; current embedded-compatible path stores `None` |
| `data/quantized_weights.npz` | Quantized int8/i32 weights |
| `firmware/esp32-rust/src/model_weights.rs` | Rust constants generated from quantized weights |
| `data/esp32_predictions.csv` | UART or dry-run prediction log |
| `data/comparison_report.csv` | PC vs ESP32/int8 comparison |
| `data/optimization_targets.json` | Optimization summary |

## Hardware Target

The firmware is configured for ESP32-S3:

| Signal | ESP32-S3 GPIO | Notes |
|---|---:|---|
| ECG analog input | GPIO4 | ADC1 input in ADC mode |
| LED alert | GPIO2 | Built-in LED on many boards |
| Buzzer alert | GPIO21 | Used because GPIO25 is not available on ESP32-S3 |
| UART0 RX | GPIO44 | UART feed mode |
| UART0 TX | GPIO43 | UART feed mode |

The AD8232 sensor path is optional. UART-feed mode can validate the model without an ECG sensor.

## Software Requirements

- Python 3.9+
- Python packages from `requirements.txt`
- GNUPlot 5 or newer
- ESP Rust toolchain installed with `espup`
- `cargo`
- `espflash`

Install Python dependencies:

```powershell
cd /d D:\PropertiesProject-D\Kuliah\Pemkon\Product
py -m pip install -r requirements.txt
```

## Quick Start: PC-Only Pipeline

Run the full software pipeline without ESP32 hardware:

```bat
cd /d D:\PropertiesProject-D\Kuliah\Pemkon\Product
run_simulation.bat
```

Regenerate only plots:

```bat
run_all_plots.bat
```

## Retrain and Export the Model

Use this when you change data generation, features, or model code:

```bat
cd /d D:\PropertiesProject-D\Kuliah\Pemkon\Product
py python\generate_dummy_ecg.py
py python\train_simple_model.py
py python\quantize_weights.py
py python\export_rust_weights.py
```

After export, rebuild or flash the firmware so the ESP32 uses the new `model_weights.rs`.

## UART-Feed Evaluation

Dry-run, no hardware:

```bat
cd /d D:\PropertiesProject-D\Kuliah\Pemkon\Product
run_uart_eval.bat --dry-run
```

Hardware run:

```bat
run_uart_eval.bat COM16
```

Replace `COM16` with the port shown by Device Manager.

What it does:

1. Builds and flashes UART-feed firmware in hardware mode.
2. Sends `data/sample_ecg.csv` ADC values to ESP32-S3.
3. Reads back predictions.
4. Compares PC float32 and ESP32/int8 outputs.
5. Writes reports and charts.

## Firmware Build

From the firmware directory:

```powershell
cd /d D:\PropertiesProject-D\Kuliah\Pemkon\Product\firmware\esp32-rust
. $HOME\export-esp.ps1
cargo build --release --target xtensa-esp32s3-none-elf
```

UART-feed build:

```powershell
cargo build --release --target xtensa-esp32s3-none-elf --features uart-feed
```

Flash example:

```powershell
espflash flash target\xtensa-esp32s3-none-elf\release\ecg-esp32 --port COM16
```

`run_uart_eval.bat` automates this flow and uses an external Cargo target directory to reduce Windows file-lock issues.

## Documentation

Start with:

- `docs/INDEX.md`
- `docs/algorithm.md`
- `docs/uart_feed_evaluation.md`
- `docs/optimization_guide.md`

## Limitations

- Synthetic data is not a substitute for validated medical ECG datasets.
- ADC mode depends on hardware wiring and signal quality.
- The current classifier is intentionally tiny and educational.
- Real-time timing in ADC mode is approximate because the main loop uses delay-based sampling.
- UART CSV logging can become a bottleneck at 115200 baud.

## License

MIT License. See `LICENSE`.
