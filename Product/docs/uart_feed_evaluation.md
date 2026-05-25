# UART-Feed Evaluation

UART-feed evaluation validates the embedded int8 model using controlled PC data. It is the main way to compare PC float32 behavior with ESP32-S3 firmware behavior.

## Purpose

The UART-feed pipeline answers:

```text
How close is the ESP32/int8 model to the PC float32 model on the same input samples?
```

It avoids sensor variability by sending the same labeled synthetic samples to the PC model and the ESP32-S3.

## Modes

| Mode | Command | Hardware needed |
|---|---|---|
| Dry-run | `run_uart_eval.bat --dry-run` | No |
| Hardware | `run_uart_eval.bat COM16` | ESP32-S3 |

Dry-run simulates ESP32/int8 inference from `data/quantized_weights.npz`. Hardware mode builds, flashes, sends samples over UART, and captures ESP32 responses.

## Protocol

PC to ESP32:

```text
2048\n
```

ESP32 to PC:

```text
-1\n   buffer not full yet
0\n    normal
1\n    abnormal
```

UART-feed firmware uses UART0:

| Signal | ESP32-S3 GPIO |
|---|---:|
| RX | GPIO44 |
| TX | GPIO43 |

## One-Command Evaluation

Dry-run:

```bat
cd /d D:\PropertiesProject-D\Kuliah\Pemkon\Product
run_uart_eval.bat --dry-run
```

Hardware:

```bat
run_uart_eval.bat COM16
```

The batch file performs:

1. Build and flash firmware in hardware mode.
2. Run `uart_feed_evaluator.py`.
3. Run `compare_models.py`.
4. Run `optimization_report.py`.
5. Generate GNUPlot evaluation charts.

## Manual Hardware Flow

Build and flash UART-feed firmware:

```powershell
cd /d D:\PropertiesProject-D\Kuliah\Pemkon\Product\firmware\esp32-rust
. $HOME\export-esp.ps1
cargo build --release --target xtensa-esp32s3-none-elf --features uart-feed
espflash flash target\xtensa-esp32s3-none-elf\release\ecg-esp32 --port COM16
```

Run evaluator:

```powershell
cd /d D:\PropertiesProject-D\Kuliah\Pemkon\Product
py python\uart_feed_evaluator.py --port COM16 --baud 115200
```

Compare:

```powershell
py python\compare_models.py
py python\optimization_report.py
```

## Output Files

| File | Purpose |
|---|---|
| `data/esp32_predictions.csv` | Per-sample PC and ESP32 predictions |
| `data/comparison_report.csv` | Aligned PC/int8/ground-truth comparison |
| `data/optimization_targets.json` | Summary metrics and recommendations |
| `images/model_esp32/*.png` | ESP32/int8 charts |
| `images/evaluation/*.png` | Comparison charts |

## Current Dry-Run Result

| Metric | Value |
|---|---:|
| PC float32 accuracy | 96.5% |
| ESP32/int8 dry-run accuracy | 95.4% |
| Quantization delta | 1.05% |
| Agreement | 98.9% |
| Disagreements | 25 out of 2373 valid predictions |

## Troubleshooting

| Symptom | Check |
|---|---|
| Cannot open COM port | Close Serial Monitor, PuTTY, espflash monitor, or other terminal |
| Mostly `-1` predictions | Buffer has not filled, or ESP32 is not returning valid lines |
| Timeouts | Wrong port, firmware not flashed in UART-feed mode, board reset |
| PC and ESP32 disagree heavily | Check feature extraction, quantization export, and Rust weight layout |
| Batch fails at GNUPlot | Confirm `gnuplot\evaluation\*.gp` exists and GNUPlot is installed |

## Notes

- Hardware round-trip latency is not the same as on-device inference time.
- UART at 115200 baud can be slow for dense per-sample evaluation.
- Dry-run is useful for model correctness; hardware mode is needed to validate flashing and UART behavior.
