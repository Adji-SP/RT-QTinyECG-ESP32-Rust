# System Block Diagram

This document describes the signal flow architecture of the RT-QTinyECG-ESP32-Rust system.

---

## Full System Block Diagram

```mermaid
flowchart LR
    subgraph HW["Hardware Layer"]
        A["🫀 ECG Sensor\nAD8232 Module\nor Simulated Signal"]
        B["⚡ ADC Input\nGPIO34 / ADC1_CH6\n12-bit, 0–4095"]
    end

    subgraph FW["ESP32 Firmware (Rust no_std)"]
        C["⏱️ Timer Sampling\n250 Hz Fixed Rate\n4 ms interval"]
        D["🔄 Ring Buffer\n128 samples\n≈ 512 ms window"]
        E["📉 Moving Average\nFilter\nWindow: 8 samples"]
        F["🧮 Feature Extraction\nmean, max, min\npeak-to-peak, energy"]
        G["🤖 Quantized Inference\nThreshold Classifier\nor Int8 Tiny MLP"]
        H["⚖️ Decision Logic\n0 = Normal\n1 = Abnormal"]
    end

    subgraph OUT["Output Layer"]
        I["🔴 LED Alert\nGPIO2\nON if Abnormal"]
        J["🔊 Buzzer Alert\nGPIO25\nON if Abnormal"]
        K["📡 UART Logger\n115200 baud\nCSV format"]
        L["📊 GNUPlot\nVisualization\nPNG Charts"]
    end

    A -->|"Analog ECG Signal\n0–3.3V"| B
    B -->|"ADC Sample\n0–4095"| C
    C -->|"Raw ADC Value\n@ 250 Hz"| D
    D -->|"Window of 128 Samples"| E
    E -->|"Filtered Sample\n(per-sample)"| D
    E -->|"Filtered Window\n(when full)"| F
    F -->|"5 Features\n[mean,max,min,p2p,E]"| G
    G -->|"Prediction\n0 or 1"| H
    H -->|"Alert ON"| I
    H -->|"Alert ON"| J
    H -->|"Log CSV Line"| K
    K -->|"CSV File"| L
```

---

## PC Simulation Block Diagram

```mermaid
flowchart LR
    subgraph PY["Python Simulation (Approach B)"]
        P1["📁 generate_dummy_ecg.py\nSynthetic ECG\n250 Hz, 10 sec"]
        P2["📄 sample_ecg.csv\nRaw + Labels"]
        P3["🔁 realtime_ecg_simulator.py\nMirrors ESP32 Pipeline"]
        P4["📊 simulated_realtime_log.csv\nFull log with metrics"]
        P5["📈 metrics.py\nAcc, Prec, Recall, F1\nLatency, Stability"]
        P6["🎨 GNUPlot Scripts\n3 × PNG charts"]
    end

    subgraph ML["Model Training (Optional)"]
        M1["🏋️ train_simple_model.py\nLogistic/MLP\nSklearn"]
        M2["🔢 quantize_weights.py\nFloat32 → Int8"]
        M3["🦀 export_rust_weights.py\n→ model_weights.rs"]
    end

    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5
    P4 --> P6
    P2 --> M1
    M1 --> M2
    M2 --> M3
```

---

## Memory Architecture (ESP32)

```mermaid
block-beta
    columns 3
    block:flash["Flash (4 MB)"]:1
        fw["Firmware Binary\n~50-150 KB"]
        weights["Model Weights\n~100 bytes"]
    end
    block:iram["IRAM (128 KB)"]:1
        code["Rust Code\n~10-30 KB"]
        isr["ISR Handlers\n~1 KB"]
    end
    block:dram["DRAM (520 KB)"]:1
        ringbuf["Ring Buffer\n512 bytes"]
        filter["Filter State\n32 bytes"]
        logbuf["Log Buffer\n64 bytes"]
        stack["Stack\n~4 KB"]
    end
```

---

## Signal Processing Pipeline Detail

| Stage | Input | Output | Latency |
|---|---|---|---|
| ADC Read | Analog voltage | 12-bit integer | ~20–100 µs |
| Moving Avg | New sample | Filtered sample | ~1 µs |
| Ring Buffer | Filtered sample | Window of 128 | O(1), ~0.1 µs |
| Feature Extract | 128-sample window | 5 features | ~5 µs |
| Threshold Classify | 5 features | 0 or 1 | ~1 µs |
| MLP Classify | 5 features | 0 or 1 | ~10–50 µs |
| GPIO Alert | Prediction | LED/Buzzer ON | ~2 µs |
| UART Log | CSV line | Serial TX | ~5 ms @ 115200 |

**Total computational latency (excl. UART): ~30–160 µs**
**Sampling interval: 4000 µs** → large safety margin.
