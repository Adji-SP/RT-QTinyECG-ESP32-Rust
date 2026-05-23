# Wiring Guide: ESP32 + AD8232 ECG Module

> ⚠️ **SAFETY NOTE**: This is an educational prototype only.
> The AD8232 module is NOT a certified medical device.
> DO NOT use this system for clinical diagnosis, medical monitoring, or patient care.
> Keep the system isolated from the body as much as possible.
> Use with 3.3V only (do NOT connect 5V to analog input).

---

## Component List

| Component | Quantity | Notes |
|---|---|---|
| ESP32 DevKit V1 | 1 | Any generic ESP32 board |
| AD8232 ECG Module | 1 | Includes 3-lead cable |
| ECG Electrode Stickers | 3 | Disposable stick-on electrodes |
| LED (any color) | 1 | Optional (GPIO2 = built-in) |
| Active Buzzer | 1 | 3.3V compatible, active type |
| 1kΩ Resistor | 1 | Optional: buzzer current limiting |
| Breadboard | 1 | For prototyping |
| Jumper Wires | ~10 | Female-to-male for AD8232 |
| USB Cable | 1 | For ESP32 power + programming |

---

## Pin Connections

### AD8232 → ESP32

| AD8232 Pin | ESP32 Pin | Wire Color (suggested) | Notes |
|---|---|---|---|
| `3.3V` or `VCC` | `3V3` | Red | Power supply (3.3V ONLY) |
| `GND` | `GND` | Black | Common ground |
| `OUTPUT` | `GPIO34` | Yellow | ECG analog signal → ADC input |
| `LO+` | `GPIO32` | Orange | Lead-off detect + |
| `LO-` | `GPIO33` | Blue | Lead-off detect - |
| `SDN` | Not connected | – | Shutdown pin; float = normal operation |

### Alert Outputs

| Component | ESP32 GPIO | Notes |
|---|---|---|
| LED (+ to GPIO, – to GND via 330Ω) | `GPIO2` | Built-in LED on DevKit V1 |
| Active Buzzer (+ to GPIO, – to GND) | `GPIO25` | 3.3V active buzzer |

---

## Wiring Diagram (ASCII)

```
ESP32 DevKit V1
┌──────────────────────────────┐
│  3V3  ────────────────────── │──── AD8232 VCC (RED)
│  GND  ────────────────────── │──── AD8232 GND (BLACK)
│                              │
│  GPIO34 (ADC1_CH6, INPUT)── │──── AD8232 OUTPUT (YELLOW)
│  GPIO32 (LO+) ───────────── │──── AD8232 LO+ (ORANGE)
│  GPIO33 (LO-) ───────────── │──── AD8232 LO- (BLUE)
│                              │
│  GPIO2  (LED, OUTPUT) ────── │──── LED (+ pole) → 330Ω → GND
│  GPIO25 (BUZZER, OUTPUT) ─── │──── Active Buzzer (+) → Buzzer → GND
│                              │
│  USB (UART0) ─────────────── │──── PC (Serial Monitor / CSV capture)
└──────────────────────────────┘
```

---

## AD8232 Electrode Placement (Educational Demo Only)

The AD8232 is designed for 3-lead ECG placement:

```
Left Arm (LA) ────── Yellow/Green electrode
Right Arm (RA) ───── Red electrode
Right Leg (RL) ───── Black electrode (reference/drive)
```

For educational demonstration:
- Wrist placement is common but not clinically accurate.
- Chest placement gives a clearer ECG waveform.
- Ensure all electrodes make good skin contact.
- Avoid movement (motion artifacts corrupt the signal).

---

## Critical Wiring Rules

1. **ADC input must NOT exceed 3.3V**
   - GPIO34 ADC input maximum is 3.3V
   - AD8232 powered at 3.3V will not exceed this
   - Never connect 5V signals to GPIO34

2. **Use common ground**
   - Connect AD8232 GND and ESP32 GND together
   - Without common ground, the ADC reads noise, not ECG

3. **GPIO34, 35, 36, 39 are INPUT-ONLY**
   - These ESP32 pins have no internal pullup/pulldown
   - They can only be used as inputs, not outputs
   - Do not accidentally use them for LED or buzzer

4. **GPIO25 and DAC2**
   - GPIO25 doubles as ESP32 DAC2 output
   - Using it as digital output (buzzer) is fine
   - If you need DAC2 for something else, move buzzer to GPIO26

5. **Buzzer current**
   - ESP32 GPIO can source ~12 mA max
   - For louder buzzers (>20 mA), use an NPN transistor:
     - GPIO → 1kΩ → NPN base
     - NPN collector → Buzzer → 3.3V
     - NPN emitter → GND

---

## Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| ADC always reads 0 | Lead-off detected | Check electrode connections |
| ADC always reads 4095 | Floating input | Check GND connection |
| Noisy signal | Motion artifact | Sit still, press electrodes firmly |
| No serial output | Wrong baud rate | Set terminal to 115200 |
| Firmware won't flash | Wrong port | Check Device Manager (Windows) |
| LED stays ON | Algorithm tuning | Adjust thresholds in inference.rs |

---

## Safety Reminder

> This project is for **educational and learning purposes only**.
> It is not a medical device and must not be used for clinical monitoring.
> If you experience any cardiac symptoms, consult a licensed physician.
