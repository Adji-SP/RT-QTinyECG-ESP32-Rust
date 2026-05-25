# System Block Diagram

This document describes the current system architecture.

## Full System

```mermaid
flowchart LR
    subgraph PC["PC / Python"]
        A["generate_dummy_ecg.py"]
        B["sample_ecg.csv"]
        C["train_simple_model.py"]
        D["quantize_weights.py"]
        E["export_rust_weights.py"]
        F["uart_feed_evaluator.py"]
        G["compare_models.py"]
        H["optimization_report.py"]
    end

    subgraph FW["ESP32-S3 Rust Firmware"]
        I["UART-feed mode\nor ADC mode"]
        J["Moving average\nN = 8"]
        K["Ring buffer\n128 samples"]
        L["Feature extraction\n5 inputs"]
        M["Int8 MLP\n5 -> 8 -> 1"]
        N["LED GPIO2\nBuzzer GPIO21"]
    end

    subgraph OUT["Outputs"]
        O["esp32_predictions.csv"]
        P["comparison_report.csv"]
        Q["optimization_targets.json"]
        R["PNG charts"]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> M
    B --> F
    F --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    M --> F
    F --> O
    O --> G
    B --> G
    G --> P
    P --> H
    H --> Q
    P --> R
```

## Firmware Data Path

```mermaid
flowchart LR
    A["ADC GPIO4\nor UART0 RX GPIO44"] --> B["Moving average\n8 samples"]
    B --> C["Ring buffer\n128 filtered samples"]
    C --> D["Feature extraction"]
    D --> E["Feature normalization\nint8-like range"]
    E --> F["Quantized MLP\nint8 weights, i32 accum"]
    F --> G{"prediction"}
    G -->|"0 normal"| H["LED low\nBuzzer low"]
    G -->|"1 abnormal"| I["LED high\nBuzzer high"]
    F --> J["UART log or prediction reply"]
```

## Training and Export Path

```mermaid
flowchart LR
    A["sample_ecg.csv"] --> B["moving average"]
    B --> C["128-sample windows\nfirmware cadence"]
    C --> D["firmware-compatible features"]
    D --> E["sklearn MLP\n5 -> 8 -> 1"]
    E --> F["model.pkl"]
    F --> G["quantized_weights.npz"]
    G --> H["model_weights.rs"]
```

## File Outputs

| Output | Source |
|---|---|
| `data/model.pkl` | `train_simple_model.py` |
| `data/quantized_weights.npz` | `quantize_weights.py` |
| `firmware/esp32-rust/src/model_weights.rs` | `export_rust_weights.py` |
| `data/esp32_predictions.csv` | `uart_feed_evaluator.py` |
| `data/comparison_report.csv` | `compare_models.py` |
| `data/optimization_targets.json` | `optimization_report.py` |
| `images/evaluation/*.png` | GNUPlot scripts |

## Memory View

```mermaid
block-beta
    columns 3
    block:flash["Flash"]:1
        fw["Firmware binary"]
        weights["Int8 model weights"]
    end
    block:ram["RAM"]:1
        ring["Ring buffer: 512 B"]
        filt["Filter state: 32 B"]
        stack["Stack frames"]
    end
    block:pc["PC artifacts"]:1
        model["model.pkl"]
        npz["quantized_weights.npz"]
        reports["CSV/JSON reports"]
    end
```

