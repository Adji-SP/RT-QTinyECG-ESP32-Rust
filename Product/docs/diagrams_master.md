# RT-QTinyECG-ESP32-Rust — Master Diagram Reference

> All diagrams in one place. Every diagram here is the canonical source of truth; individual doc files contain the same diagrams with surrounding prose. Navigate by section or jump directly to any diagram via the TOC.

---

## Table of Contents

| # | Section | Diagram type |
|:---:|---|---|
| 1 | [Full System Architecture](#1-full-system-architecture) | `flowchart LR` |
| 2 | [Three-Layer System Overview](#2-three-layer-system-overview) | `graph TD` |
| 3 | [Firmware Data Path (sample → alert)](#3-firmware-data-path) | `flowchart TD` |
| 4 | [Training → Deployment Pipeline](#4-training--deployment-pipeline) | `flowchart LR` |
| 5 | [Two Evaluation Paths](#5-two-evaluation-paths) | `flowchart TD` |
| 6 | [Firmware Boot Sequence](#6-firmware-boot-sequence) | `flowchart TD` |
| 7 | [ADC Mode Main Loop](#7-adc-mode-main-loop) | `flowchart TD` |
| 8 | [UART-Feed Mode Main Loop](#8-uart-feed-mode-main-loop) | `flowchart TD` |
| 9 | [Quantized MLP Inference Pipeline](#9-quantized-mlp-inference-pipeline) | `flowchart LR` |
| 10 | [MLP Neural Network Architecture](#10-mlp-neural-network-architecture) | `graph LR` |
| 11 | [Threshold Classifier Decision Tree](#11-threshold-classifier-decision-tree) | `flowchart TD` |
| 12 | [Signal Processing Chain (DSP)](#12-signal-processing-chain) | `flowchart LR` |
| 13 | [Ring Buffer State Machine](#13-ring-buffer-state-machine) | `stateDiagram-v2` |
| 14 | [Alert Output State Machine](#14-alert-output-state-machine) | `stateDiagram-v2` |
| 15 | [UART-Feed Evaluation Workflow](#15-uart-feed-evaluation-workflow) | `flowchart TD` |
| 16 | [UART Communication Sequence](#16-uart-communication-sequence) | `sequenceDiagram` |
| 17 | [Sampling Timing Diagram](#17-sampling-timing-diagram) | `sequenceDiagram` |
| 18 | [Per-Sample Timing Budget (Gantt)](#18-per-sample-timing-budget) | `gantt` |
| 19 | [Alert Latency Timeline](#19-alert-latency-timeline) | `timeline` |
| 20 | [Optimization Improvement History](#20-optimization-improvement-history) | `timeline` |
| 21 | [Optimization Decision Tree](#21-optimization-decision-tree) | `flowchart TD` |
| 22 | [Full Retraining Loop](#22-full-retraining-loop) | `flowchart TD` |
| 23 | [Fine-Tuning Safety Flow](#23-fine-tuning-safety-flow) | `flowchart TD` |
| 24 | [Architecture Change Checklist](#24-architecture-change-checklist) | `flowchart LR` |
| 25 | [Component Dependency Graph](#25-component-dependency-graph) | `graph TD` |
| 26 | [Memory Layout](#26-memory-layout) | `block-beta` |
| 27 | [ECG Signal Flow (Electrodes → Alert)](#27-ecg-signal-flow-electrodes--alert) | `flowchart TD` |
| 28 | [AD8232 Wiring Diagram](#28-ad8232-wiring-diagram) | `graph LR` |
| 29 | [Buzzer Driver Circuit](#29-buzzer-driver-circuit) | `graph TD` |
| 30 | [Power Supply Chain](#30-power-supply-chain) | `flowchart LR` |
| 31 | [UART Edge Case Handling](#31-uart-edge-case-handling) | `flowchart TD` |
| 32 | [Failure Diagnosis Flowchart](#32-failure-diagnosis-flowchart) | `flowchart TD` |

---

## 1. Full System Architecture

End-to-end view: PC toolchain → firmware → outputs.

```mermaid
flowchart LR
    subgraph PC["💻 PC / Python Toolchain"]
        A["generate_dummy_ecg.py\n↓ synthetic ECG-like data"]
        B["sample_ecg.csv\n(labeled waveform)"]
        C["train_simple_model.py\n↓ sklearn MLP float32"]
        D["quantize_weights.py\n↓ int8 symmetric quant"]
        E["export_rust_weights.py\n↓ const Rust arrays"]
        F["uart_feed_evaluator.py\n↓ UART or dry-run eval"]
        G["compare_models.py\n↓ PC vs ESP32 alignment"]
        H["optimization_report.py\n↓ metrics + targets"]
    end

    subgraph FW["⚡ ESP32-S3 Rust Firmware (no_std)"]
        I["UART-feed mode\nor ADC mode\n(GPIO4 / GPIO44)"]
        J["Moving average\nN = 8 samples"]
        K["Ring buffer\n128 × i32 (512 B)"]
        L["Feature extraction\n5 aggregate features"]
        M["Int8 MLP\n5 → 8 → 1  (~65 MACs)"]
        N["Alert outputs\nLED GPIO2 / Buzzer GPIO21"]
    end

    subgraph OUT["📊 Outputs"]
        O["esp32_predictions.csv"]
        P["comparison_report.csv"]
        Q["optimization_targets.json"]
        R["PNG evaluation charts"]
    end

    A --> B --> C --> D --> E
    E -->|"model_weights.rs\n(embedded weights)"| M
    B --> F
    F -->|"ADC integers\nvia UART 115200"| I
    I --> J --> K --> L --> M --> N
    M -->|"prediction reply\n-1 / 0 / 1"| F
    F --> O --> G
    B --> G --> P --> H --> Q
    P --> R
```

---

## 2. Three-Layer System Overview

PC Layer / Firmware Layer / Hardware Layer.

```mermaid
graph TD
    subgraph PC_Layer["💻 PC Layer — Python"]
        GEN["generate_dummy_ecg.py\n(synthetic ECG-like data)"]
        TRAIN["train_simple_model.py\n(sklearn MLP, float32)"]
        QUANT["quantize_weights.py\n(int8 symmetric quantization)"]
        EXPORT["export_rust_weights.py\n(→ model_weights.rs)"]
        EVAL["uart_feed_evaluator.py\n(UART evaluation)"]
        COMPARE["compare_models.py\n(PC vs ESP32 comparison)"]
        OPT["optimization_report.py\n(metrics + targets)"]
    end

    subgraph FW_Layer["⚡ Firmware Layer — Embedded Rust (no_std)"]
        SAMPLE["Sample Acquisition\nADC GPIO4 or UART0 RX"]
        FILTER["Moving Average Filter\nN = 8 samples"]
        BUFFER["Ring Buffer\n128 × i32 (512 ms)"]
        FEATURE["Feature Extraction\n5 features"]
        INFER["Int8 MLP Inference\n5 → 8 → 1"]
        ALERT["Alert Outputs\nLED GPIO2 / Buzzer GPIO21"]
    end

    subgraph HW_Layer["🔬 Hardware Layer"]
        AD8232["AD8232 ECG Module\n(analog front-end)"]
        ESP32S3["ESP32-S3\n240 MHz, 12-bit ADC"]
    end

    GEN --> TRAIN --> QUANT --> EXPORT
    EXPORT -->|"model_weights.rs"| INFER
    EVAL -->|"CSV samples via UART"| SAMPLE
    SAMPLE --> FILTER --> BUFFER --> FEATURE --> INFER --> ALERT
    INFER -->|"prediction reply"| EVAL
    EVAL --> COMPARE --> OPT
    AD8232 -->|"analog ECG voltage"| ESP32S3
    ESP32S3 -->|"ADC reads"| SAMPLE
```

---

## 3. Firmware Data Path

Traces one sample from ADC/UART acquisition to GPIO alert output.

```mermaid
flowchart TD
    subgraph ACQ["📥 Acquisition"]
        ADC["ADC1 CH3\nGPIO4 (ADC mode)\n12-bit: 0–4095"]
        UART_IN["UART0 RX\nGPIO44 (UART-feed)\nASCII integer\\n"]
    end

    subgraph FILTER["🔧 Moving Average Filter"]
        MA_BUF["Filter state\n[i32; 8]\n(32 bytes static)"]
        MA_OUT["filtered_sample\n(i32)"]
    end

    subgraph BUFFER["💾 Ring Buffer"]
        RB["[i32; 128]\n512 bytes static\nhead pointer (usize)"]
        CHECK{"count ≥ 128?"}
    end

    subgraph FEATURES["📐 Feature Extraction"]
        F_MEAN["mean = Σx / 128"]
        F_MAX["max = max(x[0..128])"]
        F_MIN["min = min(x[0..128])"]
        F_P2P["peak_to_peak = max − min"]
        F_ENERGY["energy = Σ(x²) / 128 / 4096"]
    end

    subgraph NORM["⚖️ Normalization"]
        NORM_STEP["feat_max = max(|feat[i]|, 1)\nfeat_q[i] = clamp(feat[i]×127÷feat_max, −128, 127)"]
    end

    subgraph INFER["🧠 Int8 MLP"]
        L1["Layer 1: W1[8×5] × feat_q + B1\n→ ReLU → re-quantize hidden"]
        L2["Layer 2: W2[8] × hidden_q + B2"]
        DEC{"output > 0?"}
    end

    subgraph ALERT["🔔 Alert + Log"]
        GPIO_ON["LED GPIO2 HIGH\nBuzzer GPIO21 HIGH"]
        GPIO_OFF["LED GPIO2 LOW\nBuzzer GPIO21 LOW"]
        UART_OUT["UART0 TX GPIO43\n→ PC: prediction reply"]
    end

    ADC --> MA_BUF
    UART_IN --> MA_BUF
    MA_BUF --> MA_OUT --> RB --> CHECK
    CHECK -->|"No (count < 128)\nprediction = -1"| UART_OUT
    CHECK -->|"Yes"| F_MEAN & F_MAX & F_MIN
    F_MAX & F_MIN --> F_P2P
    F_MEAN & F_MAX & F_MIN & F_P2P --> F_ENERGY
    F_MEAN & F_MAX & F_MIN & F_P2P & F_ENERGY --> NORM_STEP
    NORM_STEP --> L1 --> L2 --> DEC
    DEC -->|"1 = Abnormal"| GPIO_ON
    DEC -->|"0 = Normal"| GPIO_OFF
    GPIO_ON & GPIO_OFF --> UART_OUT
```

---

## 4. Training → Deployment Pipeline

How a Python model becomes embedded Rust const arrays.

```mermaid
flowchart LR
    subgraph Data["📊 Data Generation"]
        CSV["sample_ecg.csv\n(synthetic ECG-like\nwaveforms + labels)"]
    end

    subgraph Training["🏋️ Training"]
        T1["Feature extraction\n(firmware-compatible)"]
        T2["MLP training\n5→8→1, float32"]
        T3["model.pkl"]
    end

    subgraph Quantization["🔢 Quantization"]
        Q1["quantize_weights.py"]
        Q2["quantized_weights.npz\n(int8 W, i32 B, scale)"]
    end

    subgraph Export["📤 Rust Export"]
        E1["export_rust_weights.py\n(transposes W1)"]
        E2["model_weights.rs\n(const arrays)"]
    end

    subgraph Firmware["⚡ ESP32-S3"]
        F1["cargo build --release"]
        F2["ECG firmware binary\n+ embedded weights"]
    end

    subgraph Evaluation["📈 Evaluation"]
        EV1["uart_feed_evaluator.py"]
        EV2["compare_models.py"]
        EV3["optimization_report.py"]
    end

    CSV --> T1 --> T2 --> T3
    T3 --> Q1 --> Q2
    Q2 --> E1 --> E2
    E2 --> F1 --> F2
    CSV -->|"validation samples"| EV1
    F2 -->|"UART predictions"| EV1
    Q2 -->|"dry-run simulation"| EV1
    EV1 --> EV2 --> EV3
```

---

## 5. Two Evaluation Paths

Dry-run (no hardware) vs hardware UART evaluation.

```mermaid
flowchart TD
    START["sample_ecg.csv\n(validation data)"] --> CHOICE{"Evaluation mode?"}

    CHOICE -->|"--dry-run\n(no hardware)"| DRY["Python simulates\nint8 ESP32 path\nfrom quantized_weights.npz"]
    CHOICE -->|"COM16\n(hardware)"| HW["Send samples via UART\nto real ESP32-S3\nCapture predictions"]

    DRY --> OUT["esp32_predictions.csv"]
    HW --> OUT
    OUT --> COMP["compare_models.py\n→ comparison_report.csv"]
    COMP --> RPT["optimization_report.py\n→ optimization_targets.json"]
```

---

## 6. Firmware Boot Sequence

Startup initialization before entering the main sample loop.

```mermaid
flowchart TD
    BOOT([" 🟢 Boot / Reset"])
    INIT_GPIO["Initialize GPIOs\nGPIO2 → Output (LED)\nGPIO21 → Output (Buzzer)\nGPIO4 → Input (ADC)\nGPIO43/44 → UART0"]
    INIT_BUF["Zero ring buffer\n[i32; 128]\nhead = 0, count = 0"]
    INIT_FILT["Zero MA filter state\n[i32; 8]\nrunning_sum = 0"]
    CHECK_MODE{"Feature flag:\nuart-feed?"}
    ADC_LOOP["Enter ADC main loop\n(see diagram 7)"]
    UART_LOOP["Enter UART-feed main loop\n(see diagram 8)"]

    BOOT --> INIT_GPIO --> INIT_BUF --> INIT_FILT --> CHECK_MODE
    CHECK_MODE -->|"No (ADC mode)"| ADC_LOOP
    CHECK_MODE -->|"Yes (UART-feed)"| UART_LOOP
```

---

## 7. ADC Mode Main Loop

Real-time ECG capture loop at 250 Hz from GPIO4.

```mermaid
flowchart TD
    START(["ADC Loop"])
    DELAY["⏱ Wait 4 ms\n(250 Hz cadence)"]
    READ["Read ADC1 CH3\nGPIO4 → raw_sample: i32\n(0 – 4095)"]
    FILTER["Moving average filter:\nrunning_sum += raw − oldest\nfiltered = running_sum / 8"]
    PUSH["Push filtered sample\nto ring buffer (head)\nhead = (head+1) % 128\ncount = min(count+1, 128)"]
    FULL{"count == 128?"}
    EXTRACT["Extract 5 features\nfrom ring buffer"]
    NORMALIZE["Normalize features\nper-window int8 scale"]
    INFER["Run int8 MLP\nW1·feat_q + B1 → ReLU → W2·hidden_q + B2"]
    DECIDE{"prediction > 0?"}
    ALERT_ON["GPIO2 HIGH (LED ON)\nGPIO21 HIGH (Buzzer ON)\nUART log: '1\\n'"]
    ALERT_OFF["GPIO2 LOW (LED OFF)\nGPIO21 LOW (Buzzer OFF)\nUART log: '0\\n'"]
    NOT_READY["UART log: '0\\n'\n(buffer filling)"]
    TICK["Increment timestamp"]

    START --> DELAY --> READ --> FILTER --> PUSH --> FULL
    FULL -->|"No"| NOT_READY --> TICK --> DELAY
    FULL -->|"Yes"| EXTRACT --> NORMALIZE --> INFER --> DECIDE
    DECIDE -->|"Yes (abnormal)"| ALERT_ON --> TICK
    DECIDE -->|"No (normal)"| ALERT_OFF --> TICK
```

---

## 8. UART-Feed Mode Main Loop

Controlled evaluation loop: receives sample integers from PC via UART.

```mermaid
flowchart TD
    START(["UART-Feed Loop"])
    RECV["Block on UART0 RX GPIO44\nRead ASCII integer + '\\n'"]
    PARSE["Parse ASCII → i32\nraw_sample"]
    FILTER["Moving average filter\nrunning_sum update"]
    PUSH["Push to ring buffer"]
    FULL{"count == 128?"}
    EXTRACT["Extract 5 features"]
    NORMALIZE["Normalize per-window"]
    INFER["Int8 MLP inference"]
    DECIDE{"prediction > 0?"}
    REPLY_ABN["UART0 TX GPIO43:\n'1\\n'\nGPIO2 HIGH, GPIO21 HIGH"]
    REPLY_NRM["UART0 TX GPIO43:\n'0\\n'\nGPIO2 LOW, GPIO21 LOW"]
    REPLY_FILL["UART0 TX GPIO43:\n'-1\\n'\n(buffer filling)"]

    START --> RECV --> PARSE --> FILTER --> PUSH --> FULL
    FULL -->|"No"| REPLY_FILL --> RECV
    FULL -->|"Yes"| EXTRACT --> NORMALIZE --> INFER --> DECIDE
    DECIDE -->|"Yes"| REPLY_ABN --> RECV
    DECIDE -->|"No"| REPLY_NRM --> RECV
```

---

## 9. Quantized MLP Inference Pipeline

Complete step-by-step inference from window to prediction.

```mermaid
flowchart LR
    subgraph INPUT["📥 Input"]
        WIN["Ring buffer window\n128 × i32 samples"]
    end

    subgraph FEAT["📐 Feature Extraction"]
        F1["mean = Σx/128"]
        F2["max = max(x)"]
        F3["min = min(x)"]
        F4["p2p = max − min"]
        F5["energy = Σ(x²)/128/4096"]
    end

    subgraph NORM["⚖️ Normalization"]
        N1["feat_max = max(|feat[i]|, 1)"]
        N2["feat_q[i] = clamp(feat[i]×127÷feat_max, −128, 127)"]
    end

    subgraph L1["🧠 Layer 1  W1[8×5] + B1[8]"]
        L1_MAC["accum[j] = Σᵢ(W1[j×5+i] × feat_q[i]) + B1[j]"]
        L1_RELU["hidden[j] = max(accum[j], 0)  ← ReLU"]
        L1_REQT["h_max = max(|hidden[j]|, 1)\nhidden_q[j] = clamp(hidden[j]×127÷h_max, −128, 127)"]
    end

    subgraph L2["🧠 Layer 2  W2[8] + B2[1]"]
        L2_MAC["output = Σⱼ(W2[j] × hidden_q[j]) + B2[0]"]
    end

    subgraph DEC["⚡ Decision"]
        THRESH{"output > 0?"}
        ABN["Prediction = 1\n🔴 Abnormal"]
        NRM["Prediction = 0\n🟢 Normal"]
    end

    WIN --> F1 & F2 & F3
    F2 & F3 --> F4
    F1 & F2 & F3 & F4 --> F5
    F1 & F2 & F3 & F4 & F5 --> N1 --> N2
    N2 --> L1_MAC --> L1_RELU --> L1_REQT
    L1_REQT --> L2_MAC --> THRESH
    THRESH -->|"Yes"| ABN
    THRESH -->|"No"| NRM
```

---

## 10. MLP Neural Network Architecture

5 inputs → 8 hidden ReLU neurons → 1 output logit.

```mermaid
graph LR
    subgraph INPUT["Input Layer (5)"]
        I1["mean"]
        I2["max"]
        I3["min"]
        I4["peak-to-peak"]
        I5["energy"]
    end

    subgraph HIDDEN["Hidden Layer (8) — ReLU"]
        H1["h₁"]
        H2["h₂"]
        H3["h₃"]
        H4["h₄"]
        H5["h₅"]
        H6["h₆"]
        H7["h₇"]
        H8["h₈"]
    end

    subgraph OUTPUT["Output Layer (1)"]
        O1["logit"]
        PRED{"> 0?"}
        C0["0 = Normal 🟢"]
        C1["1 = Abnormal 🔴"]
    end

    I1 & I2 & I3 & I4 & I5 --> H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8
    H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8 --> O1
    O1 --> PRED
    PRED -->|"Yes"| C1
    PRED -->|"No"| C0
```

---

## 11. Threshold Classifier Decision Tree

Simple rule-based fallback classifier in `inference.rs`.

```mermaid
flowchart TD
    P2P{"peak_to_peak > 600?"}
    MEAN_H{"mean > 2350?"}
    MEAN_L{"mean < 1750?"}
    ABN["🔴 ABNORMAL\nprediction = 1"]
    NRM["🟢 NORMAL\nprediction = 0"]

    P2P -->|"Yes"| ABN
    P2P -->|"No"| MEAN_H
    MEAN_H -->|"Yes"| ABN
    MEAN_H -->|"No"| MEAN_L
    MEAN_L -->|"Yes"| ABN
    MEAN_L -->|"No"| NRM
```

---

## 12. Signal Processing Chain

Raw ADC integer → smoothed → buffered → features → normalized → MLP.

```mermaid
flowchart LR
    RAW["Raw ADC\n12-bit\n0–4095"]
    MA["Moving Avg\nN=8\n(smoothed)"]
    RB["Ring Buffer\n128 × i32\n(512 ms window)"]
    FE["Feature Extraction\n5 scalars:\nmin, max, mean,\npeak-to-peak, energy"]
    NORM["Per-Window\nNormalization\n→ int8 range"]
    MLP["Int8 MLP\n5→8→1\n~65 MACs"]
    PRED["Prediction\n0 or 1"]

    RAW -->|"N=8\nrunning sum"| MA
    MA -->|"append\nhead++ % 128"| RB
    RB -->|"when full\n(count=128)"| FE
    FE -->|"scale ×127\n÷ feat_max"| NORM
    NORM -->|"W1·x+B1\nReLU\nW2·h+B2"| MLP
    MLP -->|"> 0 ?"| PRED
```

---

## 13. Ring Buffer State Machine

Fill → Ready transitions; sliding window in steady state.

```mermaid
stateDiagram-v2
    direction LR

    [*] --> Filling : Boot\n(count = 0)

    Filling : FILLING\ncount < 128\nprediction = -1

    Ready : READY\ncount = 128\nInference every sample

    Filling --> Filling : push sample\n(count++)
    Filling --> Ready : count reaches 128
    Ready --> Ready : push sample\n(overwrite oldest)\nrun inference
```

---

## 14. Alert Output State Machine

LED + Buzzer state changes driven by MLP prediction every 4 ms.

```mermaid
stateDiagram-v2
    direction LR

    [*] --> STARTUP : Boot

    STARTUP : STARTUP\nBuffer Filling\nLED OFF · Buzzer OFF\nUART: -1

    NORMAL : NORMAL\nprediction = 0\nLED OFF · Buzzer OFF\nUART: 0

    ABNORMAL : ABNORMAL\nprediction = 1\nLED ON · Buzzer ON\nUART: 1

    STARTUP --> NORMAL : count = 128\n+ pred = 0
    STARTUP --> ABNORMAL : count = 128\n+ pred = 1
    NORMAL --> ABNORMAL : pred = 1
    ABNORMAL --> NORMAL : pred = 0
    NORMAL --> NORMAL : pred = 0
    ABNORMAL --> ABNORMAL : pred = 1
```

---

## 15. UART-Feed Evaluation Workflow

What `run_uart_eval.bat` does end-to-end.

```mermaid
flowchart TD
    CSV["sample_ecg.csv"]
    EVAL["uart_feed_evaluator.py"]
    MODE{"Mode?"}

    subgraph DRY["Dry-Run (--dry-run)"]
        DR1["Load quantized_weights.npz"]
        DR2["Python simulates int8\nESP32 inference\n(same integer arithmetic)"]
    end

    subgraph HW["Hardware (COM16)"]
        HW1["cargo build --release\n--features uart-feed"]
        HW2["espflash flash --port COM16"]
        HW3["Send sample via UART\nGPIO44 RX @ 115200"]
        HW4["Wait for reply\nGPIO43 TX: -1 / 0 / 1"]
    end

    OUT["esp32_predictions.csv"]
    COMP["compare_models.py\n→ comparison_report.csv"]
    OPT["optimization_report.py\n→ optimization_targets.json"]
    PLOTS["GNUPlot scripts\n→ PNG charts"]

    CSV --> EVAL --> MODE
    MODE -->|"--dry-run"| DR1 --> DR2 --> OUT
    MODE -->|"COM16"| HW1 --> HW2 --> HW3 --> HW4 --> OUT
    OUT --> COMP --> OPT
    COMP --> PLOTS
```

---

## 16. UART Communication Sequence

Full sample → prediction round-trip between PC and ESP32-S3.

```mermaid
sequenceDiagram
    participant CSV as sample_ecg.csv
    participant PY as uart_feed_evaluator.py
    participant UART as USB / UART0 (115200 baud)
    participant FW as ESP32-S3 Firmware

    CSV->>PY: labeled ADC sample stream

    Note over PY,FW: Startup — ring buffer filling (first 127 samples)
    loop samples 1 → 127
        PY->>UART: "2048\n"
        UART->>FW: integer via GPIO44 RX
        FW->>FW: filter → push (count < 128)
        FW->>UART: "-1\n" via GPIO43 TX
        UART->>PY: "-1" (skip in metrics)
    end

    Note over PY,FW: Steady-state inference (sample 128+)
    PY->>UART: "2060\n" (sample 128)
    UART->>FW: integer
    FW->>FW: push → extract features → MLP
    FW->>UART: "0\n" (Normal)
    UART->>PY: "0" (record valid prediction)

    PY->>UART: "3600\n" (anomalous sample)
    UART->>FW: integer
    FW->>FW: push → high P2P → MLP → abnormal
    FW->>UART: "1\n" (Abnormal)
    UART->>PY: "1" (record valid prediction)
    Note over FW: GPIO2 LED HIGH, GPIO21 Buzzer HIGH
```

---

## 17. Sampling Timing Diagram

250 Hz sample cadence showing the per-sample work cycle.

```mermaid
sequenceDiagram
    participant CLK as 4 ms Delay
    participant ADC as ADC / UART
    participant FILT as MA Filter
    participant BUF as Ring Buffer
    participant INF as Inference + GPIO

    loop Every 4 ms
        CLK->>ADC: trigger read
        ADC->>FILT: raw 12-bit integer
        FILT->>BUF: filtered sample
        alt Buffer full (count = 128)
            BUF->>INF: run features + MLP
            INF-->>CLK: GPIO update + UART reply
        else Buffer filling
            BUF-->>CLK: reply -1 (not ready)
        end
    end
```

---

## 18. Per-Sample Timing Budget

4 ms = 4000 µs budget per sample. Gantt shows time allocation.

```mermaid
gantt
    title Firmware Timing Budget per 4 ms Sample Period
    dateFormat X
    axisFormat %L µs

    section Always
    ADC read GPIO4           :  0,   50
    Moving average update    :  50,  10

    section After buffer full
    Ring buffer insert       :  60,   5
    Feature extraction (128) :  65,  15
    Feature normalization    :  80,   5
    MLP Layer 1 (40 MACs)   :  85,  30
    MLP Layer 2 (8 MACs)    :  115, 10
    GPIO alert toggle        :  125,  1

    section Optional logging
    UART TX one line         :  126, 1000
```

---

## 19. Alert Latency Timeline

Time from first sample to first valid alert output.

```mermaid
timeline
    title End-to-End Alert Latency (ADC mode, after boot)
    section One-Time Startup
        t=0 ms : Board boot + peripheral init (~50–100 ms)
    section Buffer Fill (one-time, dominates)
        t=100 ms : Sample 1 received
        t=612 ms : Sample 128 received — First inference possible
    section Steady-State (per-sample thereafter)
        t=612.00 ms : Feature extraction + MLP (~60 µs)
        t=612.06 ms : GPIO toggle — alert fires (~1 µs)
        t=613 ms : UART log line sent (~1 ms)
```

---

## 20. Optimization Improvement History

Representative timeline of model improvement during development.

```mermaid
timeline
    title Model Improvement History
    section Initial (pipeline misaligned)
        PC float32 accuracy  : ~70%
        Quantization delta   : >10% (feature mismatch)
    section After feature alignment
        PC float32 accuracy  : ~92%
        Quantization delta   : ~3%
    section Current (fully aligned)
        PC float32 accuracy  : 96.5%
        ESP32 int8 accuracy  : 95.4%
        Quantization delta   : 1.05%
        Agreement            : 98.9%
```

---

## 21. Optimization Decision Tree

What to do based on `compare_models.py` output.

```mermaid
flowchart TD
    START(["Run compare_models.py\n+ optimization_report.py"])

    START --> DELTA{"Quantization\ndelta > 5%?"}
    DELTA -->|"Yes"| FIX_DELTA["Check in order:\n1. extract_firmware_mlp_input_array() matches inference.rs\n2. W1 transposed in export_rust_weights.py?\n3. Dry-run integer math matches Rust?\n4. No scaler in embedded path?"]

    DELTA -->|"No"| ACC{"ESP32 accuracy\n< 85%?"}
    ACC -->|"Yes"| FIX_ACC["Inspect comparison_report.csv\nRetrain with more data\nIncrease max_iter"]

    ACC -->|"No"| AGREE{"Agreement\n< 95%?"}
    AGREE -->|"Yes"| FIX_AGREE["Reflash firmware\nConfirm NPZ matches RS\nCheck baud rate 115200"]

    AGREE -->|"No"| RECALL{"Recall < 0.85?"}
    RECALL -->|"Yes"| FIX_RECALL["Add abnormal patterns\nAugment dataset\nFine-tune with disagreements"]

    RECALL -->|"No"| PREC{"Precision < 0.85?"}
    PREC -->|"Yes"| FIX_PREC["Add noise/normal examples\nFine-tune with disagreements"]

    PREC -->|"No"| OK["✅ All metrics within targets\nNo optimization needed"]

    FIX_DELTA & FIX_ACC & FIX_AGREE & FIX_RECALL & FIX_PREC --> ACTION["Retrain or fine-tune\nthen re-evaluate"]
    ACTION --> START
```

---

## 22. Full Retraining Loop

Complete retrain → quantize → export → eval → hardware cycle.

```mermaid
flowchart TD
    TRIGGER(["Trigger: data / firmware / features changed"])
    GEN["py generate_dummy_ecg.py"]
    TRAIN["py train_simple_model.py\n→ model.pkl"]
    QUANT["py quantize_weights.py\n→ quantized_weights.npz"]
    EXPORT["py export_rust_weights.py\n→ model_weights.rs"]
    DRYRUN["run_uart_eval.bat --dry-run\n→ comparison_report.csv"]
    CHECK{"Metrics within\ntargets?\nacc > 90%, delta < 3%"}
    KEEP["✅ Keep model\nCommit model_weights.rs"]
    FLASH["Flash to hardware\nrun_uart_eval.bat COM16"]
    HW_CHECK{"Hardware metrics\nmatch dry-run?"}
    DONE(["✅ Done"])
    INSPECT["Inspect disagreements\ncomparison_report.csv"]
    FINETUNE["py fine_tune_model.py\n--extra-data disagreements.csv\n--augment-factor 5"]
    ROLLBACK["⚠️ Rollback: keep old model.pkl"]

    TRIGGER --> GEN --> TRAIN --> QUANT --> EXPORT --> DRYRUN --> CHECK
    CHECK -->|"Yes"| KEEP --> FLASH --> HW_CHECK
    HW_CHECK -->|"Yes"| DONE
    HW_CHECK -->|"No"| INSPECT
    CHECK -->|"No"| INSPECT
    INSPECT --> FINETUNE
    FINETUNE -->|"improved"| QUANT
    FINETUNE -->|"not improved"| ROLLBACK --> INSPECT
```

---

## 23. Fine-Tuning Safety Flow

`fine_tune_model.py` keeps the old model if fine-tuning makes things worse.

```mermaid
flowchart TD
    LOAD["Load model.pkl\n+ disagreements.csv"]
    AUGMENT["Augment disagreement samples\n× augment_factor (default 5)"]
    TRAIN["Continue training\nfrom existing weights"]
    EVAL_NEW["Evaluate fine-tuned model\non full validation set"]
    EVAL_OLD["Evaluate original model\non full validation set"]
    COMPARE{"Fine-tuned model\nbetter than original?"}
    KEEP_NEW["✅ Save fine-tuned\nmodel as model.pkl"]
    KEEP_OLD["⚠️ Keep original model.pkl\nLog: fine-tune did not improve"]
    NEXT["→ quantize_weights.py\n→ export_rust_weights.py"]

    LOAD --> AUGMENT --> TRAIN --> EVAL_NEW
    LOAD --> EVAL_OLD
    EVAL_NEW & EVAL_OLD --> COMPARE
    COMPARE -->|"Yes"| KEEP_NEW --> NEXT
    COMPARE -->|"No"| KEEP_OLD
```

---

## 24. Architecture Change Checklist

Files that must be updated together when changing MLP architecture.

```mermaid
flowchart LR
    PLAN["Decide new architecture\ne.g. 5→16→1"]
    PY["Update train_simple_model.py\nhidden_layer_sizes=(16,)"]
    EXPORT["Update export_rust_weights.py\nnew array shapes"]
    RS["Update inference.rs\nnew loop bounds + const sizes"]
    DRYRUN["Update uart_feed_evaluator.py\ndry-run simulation to match"]
    TEST["Retrain → quantize → export\n→ dry-run\nVerify delta < 3%"]

    PLAN --> PY --> EXPORT --> RS --> DRYRUN --> TEST
```

---

## 25. Component Dependency Graph

Script execution order and file dependencies.

```mermaid
graph TD
    GEN["generate_dummy_ecg.py"] -->|"writes"| CSV["sample_ecg.csv"]
    CSV -->|"reads"| TRAIN["train_simple_model.py"]
    TRAIN -->|"writes"| PKL["model.pkl"]
    PKL -->|"reads"| QUANT["quantize_weights.py"]
    QUANT -->|"writes"| NPZ["quantized_weights.npz"]
    NPZ -->|"reads"| EXPORT["export_rust_weights.py"]
    EXPORT -->|"writes"| RS["model_weights.rs"]
    RS -->|"compiled into"| FW["ESP32-S3 Firmware"]

    CSV -->|"reads"| EVAL["uart_feed_evaluator.py"]
    NPZ -->|"dry-run reads"| EVAL
    FW -->|"hardware UART"| EVAL
    EVAL -->|"writes"| PRED["esp32_predictions.csv"]

    PRED -->|"reads"| COMP["compare_models.py"]
    CSV -->|"reads"| COMP
    COMP -->|"writes"| REPORT["comparison_report.csv"]
    REPORT -->|"reads"| OPT["optimization_report.py"]
    OPT -->|"writes"| JSON["optimization_targets.json"]
    REPORT -->|"reads"| PLOT["GNUPlot scripts"]
    PLOT -->|"writes"| PNG["*.png charts"]
```

---

## 26. Memory Layout

Static memory map of the ESP32-S3 firmware at runtime.

```mermaid
block-beta
    columns 3

    block:FLASH["🗃️ Flash (SPI NOR)"]:1
        fw["Firmware binary\n(Rust .text + .rodata)\n~200–400 KB"]
        weights["model_weights.rs\nconst int8 arrays\n~84 bytes weights\n+ 16 bytes metadata"]
        boot["ESP-IDF Bootloader\n+ Partition table"]
    end

    block:RAM["🧠 SRAM (512 KB total)"]:1
        ring["Ring buffer\n[i32; 128]\n512 bytes"]
        filt["MA filter state\n[i32; 8]\n32 bytes"]
        stack["Stack + locals\n~2–4 KB"]
        free["Free SRAM\n~508 KB\n(99%+ headroom)"]
    end

    block:PC["💻 PC Artifacts"]:1
        model["model.pkl\n(float32 sklearn)"]
        npz["quantized_weights.npz\n(int8 W, i32 B, scale)"]
        reports["CSV / JSON reports"]
        images["PNG evaluation charts"]
    end
```

---

## 27. ECG Signal Flow (Electrodes → Alert)

Complete physical signal path from body to GPIO output.

```mermaid
flowchart TD
    subgraph Body["👤 Body Electrodes"]
        RA["RA — Right Arm"]
        LA["LA — Left Arm"]
        RL["RL — Right Leg (GND ref)"]
    end

    subgraph AD8232["🔬 AD8232 Analog Front-End"]
        IA["Instrumentation Amplifier\n(high CMRR, rejects common-mode)"]
        RLD_DRV["RLD Driver\n(noise reduction)"]
        HPF["High-Pass Filter\n~0.5 Hz cutoff\n(removes DC baseline wander)"]
        LPF["Low-Pass Filter\n~40 Hz cutoff\n(removes EMI / HF noise)"]
        OUT_PIN["OUTPUT pin\n≈1.65 V ± ECG swing\n(0–3.3 V range)"]
    end

    subgraph ESP32["⚡ ESP32-S3"]
        GPIO4["GPIO4 ADC1\n12-bit 0–4095"]
        DSP["MA Filter → Ring Buffer\n→ Feature Extraction\n→ Int8 MLP"]
        PRED_OUT{"Prediction\n0 or 1"}
    end

    subgraph Outputs["🔔 Alert Outputs"]
        LED["LED\nGPIO2"]
        BUZ["Active Buzzer\nGPIO21"]
        LOG["UART TX\nGPIO43"]
    end

    RA --> IA
    LA --> IA
    RL --> RLD_DRV --> IA
    IA --> HPF --> LPF --> OUT_PIN
    OUT_PIN -->|"analog voltage\n0–3.3 V"| GPIO4
    GPIO4 --> DSP --> PRED_OUT
    PRED_OUT -->|"abnormal"| LED & BUZ
    PRED_OUT --> LOG
```

---

## 28. AD8232 Wiring Diagram

Physical connections between AD8232 module and ESP32-S3 DevKit.

```mermaid
graph LR
    subgraph AD8232["AD8232 Module"]
        AD_VCC["VCC"]
        AD_GND["GND"]
        AD_OUT["OUTPUT"]
        AD_LOP["LO+"]
        AD_LOM["LO-"]
        AD_SDN["SDN"]
        AD_INP["IN+"]
        AD_INM["IN-"]
        AD_RLD["RLD"]
    end

    subgraph ESP32["ESP32-S3 DevKit"]
        E_3V3["3V3"]
        E_GND["GND"]
        E_G4["GPIO4\n(ADC1 CH3)"]
    end

    subgraph Electrodes["Body Electrodes"]
        EL_RA["RA (Right Arm)"]
        EL_LA["LA (Left Arm)"]
        EL_RL["RL (Right Leg)"]
    end

    AD_VCC -->|"🔴 3.3 V supply"| E_3V3
    AD_GND -->|"⚫ common GND"| E_GND
    AD_OUT -->|"🟡 analog ECG\n0–3.3 V"| E_G4

    EL_RA --> AD_INP
    EL_LA --> AD_INM
    EL_RL --> AD_RLD

    AD_LOP -.->|"NC — future\nlead-off detect"| NC1(("NC"))
    AD_LOM -.->|"NC — future\nlead-off detect"| NC2(("NC"))
    AD_SDN -.->|"NC — stays active"| NC3(("NC"))
```

---

## 29. Buzzer Driver Circuit

NPN transistor driver for active buzzers drawing > 5 mA.

```mermaid
graph TD
    GPIO21["GPIO21\nESP32-S3\n(control signal)"]
    R1K["1 kΩ\nBase resistor\n(limits base current ~3 mA)"]
    NPN["NPN Transistor\n2N2222 / BC547\n(switches buzzer current)"]
    V33["3.3 V\nPower rail"]
    BUZ["🔊 Active Buzzer\n(15–30 mA typical)"]
    GND["GND"]

    GPIO21 -->|"0 V / 3.3 V"| R1K
    R1K -->|"Base"| NPN
    V33 -->|"+"| BUZ
    BUZ -->|"−"| NPN
    NPN -->|"Emitter"| GND
```

---

## 30. Power Supply Chain

USB 5 V → onboard LDO → 3.3 V → peripherals.

```mermaid
flowchart LR
    USB5V["USB 5 V\n(Host PC or charger)"]
    LDO["ESP32-S3\nOnboard LDO\n3.3 V / ~500 mA"]
    AD8232_P["AD8232\n~3–5 mA"]
    LED_P["LED circuit\n~5 mA"]
    BUZ_P["Buzzer\n~20–30 mA\n(via NPN transistor)"]
    CORE["ESP32-S3 core\n~100–150 mA active"]

    USB5V -->|"via USB cable"| LDO
    LDO --> AD8232_P
    LDO --> LED_P
    LDO --> BUZ_P
    LDO --> CORE
```

---

## 31. UART Edge Case Handling

How the firmware handles malformed or out-of-range UART input.

```mermaid
flowchart TD
    RX["Received bytes\non UART0 RX GPIO44"]
    PARSE{"Parseable\nas i32?"}
    VALID{"Value in\n0 – 4095?"}
    PROCESS["Process sample normally\nthrough filter → buffer → inference"]
    CLAMP["Clamp to 0 – 4095\nlog warning"]
    ERROR["Discard line\nlog parse error\nwait for next line"]

    RX --> PARSE
    PARSE -->|"Yes"| VALID
    PARSE -->|"No (non-numeric)"| ERROR
    VALID -->|"Yes"| PROCESS
    VALID -->|"No (e.g. -999, 9999)"| CLAMP --> PROCESS
```

---

## 32. Failure Diagnosis Flowchart

Systematic debugging when evaluation metrics are out of target.

```mermaid
flowchart TD
    START["Evaluation problem detected"]

    START --> T1{"Cannot open\nCOM port?"}
    START --> T2{"Mostly -1\npredictions?"}
    START --> T3{"UART timeout\nor no reply?"}
    START --> T4{"PC and ESP32\nheavily disagree?"}
    START --> T5{"ADC reads 0\nor 4095?"}

    T1 -->|"Yes"| F1["Close all serial apps:\nSerial Monitor, PuTTY,\nespflash monitor\nThen retry"]

    T2 -->|"Yes"| F2["Count of -1 >> 127?\nFirmware in ADC mode\n→ Reflash with\n--features uart-feed"]

    T3 -->|"Yes"| F3["1. Correct COM port?\n2. Firmware flashed?\n3. Board not in reset?\n4. Baud = 115200?"]

    T4 -->|"Yes"| F4["Pipeline mismatch:\n1. W1 transposed in export?\n2. Per-window norm identical?\n3. Energy ÷4096 both sides?\n4. model_weights.rs current?"]

    T5 -->|"Yes"| F5["ADC = 0: no signal\n→ check AD8232 power\nADC = 4095: floating input\n→ check common ground"]
```

---

*Document version: 2026-06-17 | Project: RT-QTinyECG-ESP32-Rust | Firmware target: ESP32-S3 Embedded Rust (no_std)*

---

> **Cross-references:**
> - [`algorithm.md`](algorithm.md) — detailed algorithm descriptions for diagrams 3, 9, 10, 11, 12
> - [`block_diagram.md`](block_diagram.md) — system architecture for diagrams 1, 2, 25, 26
> - [`flowchart.md`](flowchart.md) — control flow for diagrams 6, 7, 8, 13, 14, 31
> - [`uart_feed_evaluation.md`](uart_feed_evaluation.md) — evaluation workflow for diagrams 15, 16, 17
> - [`real_time_design.md`](real_time_design.md) — timing rationale for diagrams 18, 19
> - [`optimization_guide.md`](optimization_guide.md) — optimization for diagrams 20, 21, 22, 23, 24
> - [`evaluation_metrics.md`](evaluation_metrics.md) — metrics for diagram 32
> - [`wiring_esp32_ad8232.md`](wiring_esp32_ad8232.md) — hardware for diagrams 27, 28, 29, 30
