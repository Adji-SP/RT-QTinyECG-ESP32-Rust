# Firmware Flowchart

This document shows the control flow of the ESP32 Rust firmware.

---

## Main Firmware Loop Flowchart

```mermaid
flowchart TD
    Start(["🚀 Start\n(Power On / Reset)"])
    Init["⚙️ Initialize ESP32 Peripherals\n- ClockControl (240 MHz)\n- GPIO (LED GPIO2, Buzzer GPIO25)\n- ADC1 (GPIO34, 11dB atten.)\n- UART0 (115200 baud)"]
    Header["📡 Print CSV Header\ntime_ms,adc_value,filtered_value,..."]
    Wait["⏱️ Wait Sampling Interval\n4 ms (delay_micros 4000)"]
    ReadADC["🔌 Read ECG ADC\nGPIO34 → 0–4095"]
    LeadOff{"🔍 Lead-off\nDetected?\n(LO+ or LO- HIGH)"}
    ReturnZero["ADC = 0\n(invalid data)"]
    Filter["📉 Moving Average Filter\npush_and_average(adc_raw)"]
    PushBuf["🔄 Push into Ring Buffer\nring_buf.push(filtered)"]
    IsFull{"📊 Ring Buffer\nFull?\n(128 samples)"}
    ExtractFeat["🧮 Extract Features\n- mean\n- max, min\n- peak_to_peak\n- energy"]
    RunInfer["🤖 Run Inference\n(Threshold or MLP)"]
    MeasureTime["⏱️ Record Inference Time\n(CCOUNT cycles → µs)"]
    IsAbnormal{"⚖️ Prediction\n== 1 (Abnormal)?"}
    AlertON["🔴 Alert ON\nLED GPIO2 HIGH\nBuzzer GPIO25 HIGH"]
    AlertOFF["🟢 Alert OFF\nLED GPIO2 LOW\nBuzzer GPIO25 LOW"]
    MeasureLatency["📏 Measure Alert Latency\n(detection → GPIO toggle)"]
    LogCSV["📡 Log CSV Line\nUART0 at 115200 baud\ntime_ms,adc,..."]
    IncTime["🕐 Increment timestamp\n+= 4 ms"]
    Loop(["🔁 Repeat"])

    Start --> Init
    Init --> Header
    Header --> Wait
    Wait --> ReadADC
    ReadADC --> LeadOff
    LeadOff -->|"YES"| ReturnZero
    LeadOff -->|"NO"| Filter
    ReturnZero --> Filter
    Filter --> PushBuf
    PushBuf --> IsFull
    IsFull -->|"NO (buffer filling)"| LogCSV
    IsFull -->|"YES (run inference)"| ExtractFeat
    ExtractFeat --> RunInfer
    RunInfer --> MeasureTime
    MeasureTime --> IsAbnormal
    IsAbnormal -->|"YES"| AlertON
    IsAbnormal -->|"NO"| AlertOFF
    AlertON --> MeasureLatency
    AlertOFF --> LogCSV
    MeasureLatency --> LogCSV
    LogCSV --> IncTime
    IncTime --> Loop
    Loop --> Wait

    style Start fill:#2d6a4f,color:#fff
    style Loop fill:#2d6a4f,color:#fff
    style AlertON fill:#7f1d1d,color:#fff
    style AlertOFF fill:#14532d,color:#fff
    style IsAbnormal fill:#92400e,color:#fff
    style IsFull fill:#1e3a5f,color:#fff
    style LeadOff fill:#4c1d95,color:#fff
```

---

## Inference Decision Flowchart (Mode A: Threshold)

```mermaid
flowchart TD
    FeatIn["📥 Input: 5 Features\n[mean, max, min, p2p, energy]"]
    R1{"peak_to_peak\n> 600 ADC?"}
    R2{"mean\n> 2350 ADC?"}
    R3{"mean\n< 1750 ADC?"}
    Abnormal(["🔴 Return 1\n(Abnormal)"])
    Normal(["🟢 Return 0\n(Normal)"])

    FeatIn --> R1
    R1 -->|"YES → high amplitude"| Abnormal
    R1 -->|"NO"| R2
    R2 -->|"YES → elevated baseline"| Abnormal
    R2 -->|"NO"| R3
    R3 -->|"YES → depressed baseline"| Abnormal
    R3 -->|"NO → all normal"| Normal

    style Abnormal fill:#7f1d1d,color:#fff
    style Normal fill:#14532d,color:#fff
```

---

## Inference Decision Flowchart (Mode B: Quantized MLP)

```mermaid
flowchart LR
    In["Input\n5 features\n(i32)"]
    Norm["Normalize to i8\n[-127, +127]"]
    L1["Layer 1\nW1 [8×5] × feat_q\n+ B1 [8]\n(i32 accumulate)"]
    ReLU["ReLU\nmax(0, x)\nper neuron"]
    ReQ["Re-quantize\nhidden → i8"]
    L2["Layer 2\nW2 [1×8] × hidden_q\n+ B2 [1]\n(i32 accumulate)"]
    Dec{"output_q > 0?"}
    Ab(["Abnormal\n(1)"])
    No(["Normal\n(0)"])

    In --> Norm
    Norm --> L1
    L1 --> ReLU
    ReLU --> ReQ
    ReQ --> L2
    L2 --> Dec
    Dec -->|"YES"| Ab
    Dec -->|"NO"| No

    style Ab fill:#7f1d1d,color:#fff
    style No fill:#14532d,color:#fff
```

---

## Python Simulator Flowchart

```mermaid
flowchart TD
    Load["📂 Load sample_ecg.csv\n2500 samples @ 250 Hz"]
    ForEach["🔁 For each sample"]
    Filt["📉 Moving Average Filter\n(8-sample window)"]
    PushR["🔄 Push to Ring Buffer\n(size 128)"]
    Full{"Buffer Full?"}
    Inf["🤖 Threshold Inference\nthreshold_inference(window)"]
    TimerInf["⏱️ Measure PC inference time\n(time.perf_counter)"]
    Alert["🔔 Update Alert State\n+ measure latency"]
    LogRow["📝 Log CSV row"]
    Done["✅ Save\nsimulated_realtime_log.csv"]

    Load --> ForEach
    ForEach --> Filt
    Filt --> PushR
    PushR --> Full
    Full -->|"NO"| LogRow
    Full -->|"YES"| Inf
    Inf --> TimerInf
    TimerInf --> Alert
    Alert --> LogRow
    LogRow --> ForEach
    ForEach -->|"All samples done"| Done
```
