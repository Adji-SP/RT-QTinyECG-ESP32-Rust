# Documentation Index

RT-QTinyECG-ESP32-Rust is an educational ESP32-S3 TinyML prototype for ECG-like signal classification. The documentation in this directory describes the current codebase and the aligned training, quantization, firmware, and evaluation flow.

> This project is not a medical device.

## Recommended Reading Order

| Document | Purpose |
|---|---|
| `../README.md` | Project overview, setup, and common commands |
| `algorithm.md` | Signal processing, feature extraction, and classifier details |
| `block_diagram.md` | End-to-end architecture diagrams |
| `flowchart.md` | Firmware and evaluation control flow |
| `uart_feed_evaluation.md` | Hardware and dry-run validation workflow |
| `evaluation_metrics.md` | Metric definitions and targets |
| `optimization_guide.md` | Retraining, fine-tuning, and optimization process |
| `real_time_design.md` | Timing, memory, and real-time design rationale |
| `wiring_esp32_ad8232.md` | Optional AD8232 wiring for ESP32-S3 ADC mode |

## Current Pipeline Summary

```text
sample_ecg.csv
    -> train_simple_model.py
    -> model.pkl
    -> quantize_weights.py
    -> quantized_weights.npz
    -> export_rust_weights.py
    -> firmware/esp32-rust/src/model_weights.rs
    -> ESP32-S3 firmware or dry-run int8 simulator
    -> compare_models.py
    -> optimization_report.py
```

## Key Commands

Run a complete PC-only simulation:

```bat
cd /d D:\PropertiesProject-D\Kuliah\Pemkon\Product
run_simulation.bat
```

Retrain and export firmware weights:

```bat
py python\generate_dummy_ecg.py
py python\train_simple_model.py
py python\quantize_weights.py
py python\export_rust_weights.py
```

Evaluate without hardware:

```bat
run_uart_eval.bat --dry-run
```

Evaluate with ESP32-S3 hardware:

```bat
run_uart_eval.bat COM16
```

Regenerate charts:

```bat
run_all_plots.bat
```

## Current Dry-Run Metrics

| Metric | Value |
|---|---:|
| PC float32 accuracy | 96.5% |
| ESP32/int8 dry-run accuracy | 95.4% |
| Quantization delta | 1.05% |
| PC/int8 agreement | 98.9% |
| Int8 model payload | 84 bytes weights/biases, plus 16 bytes scale metadata |

## Important Design Notes

- The model is trained on firmware-compatible features, not arbitrary sklearn-scaled features.
- `scaler.pkl` is retained for compatibility, but the current embedded-compatible path stores `None`.
- The first MLP layer is exported transposed because sklearn stores weights as `[features, hidden]`, while Rust reads `[hidden, features]`.
- `uart_feed_evaluator.py --dry-run` simulates the quantized int8 model from `quantized_weights.npz`.

