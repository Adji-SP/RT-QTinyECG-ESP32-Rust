# Wiring Guide: ESP32-S3 and AD8232

This guide is for optional ADC-mode experiments with an AD8232 ECG module.

> Safety notice: This project is educational only. The AD8232 breakout and this firmware are not certified medical devices. Do not use this setup for diagnosis, monitoring, or patient care. Keep the circuit USB-powered and isolated. Do not connect unknown or external mains-powered equipment to a person.

## Current Firmware Pin Map

The current firmware is configured for ESP32-S3, not classic ESP32.

| Function | ESP32-S3 GPIO | Notes |
|---|---:|---|
| ECG analog input | GPIO4 | ADC1 input |
| LED alert | GPIO2 | Built-in LED on many boards |
| Buzzer alert | GPIO21 | Active buzzer output |
| UART0 RX | GPIO44 | UART-feed mode |
| UART0 TX | GPIO43 | UART-feed mode |

Older docs and classic ESP32 examples often use GPIO34 and GPIO25. Those are not the current firmware pins.

## AD8232 Connections

| AD8232 pin | ESP32-S3 pin | Notes |
|---|---|---|
| `3.3V` / `VCC` | `3V3` | Use 3.3 V only |
| `GND` | `GND` | Common ground |
| `OUTPUT` | `GPIO4` | Analog ECG output |
| `LO+` | Not currently used by firmware | Optional future lead-off input |
| `LO-` | Not currently used by firmware | Optional future lead-off input |
| `SDN` | Not connected | Leave in normal operation state |

## Alert Outputs

| Component | ESP32-S3 GPIO | Notes |
|---|---:|---|
| LED | GPIO2 | Use built-in LED if available |
| Active buzzer | GPIO21 | Use a transistor driver for higher-current buzzers |

Recommended buzzer driver for anything above a few mA:

```text
GPIO21 -> 1 kOhm -> NPN base
Buzzer + -> 3.3 V
Buzzer - -> NPN collector
NPN emitter -> GND
```

## ASCII Wiring Sketch

```text
AD8232                  ESP32-S3
-----                   -------
VCC      -------------> 3V3
GND      -------------> GND
OUTPUT   -------------> GPIO4

LED anode ------------> GPIO2
LED cathode -> resistor -> GND

Buzzer driver input ---> GPIO21
```

## UART-Feed Mode

UART-feed mode does not require the AD8232. It receives samples from the PC over USB/UART:

```text
PC -> ESP32-S3: "2048\n"
ESP32-S3 -> PC: "-1\n", "0\n", or "1\n"
```

Use:

```bat
run_uart_eval.bat COM16
```

## Critical Electrical Rules

1. Do not feed more than 3.3 V into GPIO4.
2. Always share ground between AD8232 and ESP32-S3.
3. Avoid 5 V sensor output into ESP32 pins.
4. Use battery or USB power only for safe educational experiments.
5. Do not connect this circuit to a person while any part is connected to unsafe external equipment.

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| ADC reads 0 | No signal or wiring issue | Check AD8232 power, ground, and output |
| ADC reads 4095 | Saturated or floating input | Check ground and input voltage |
| Noisy signal | Motion artifacts or poor electrodes | Keep still, improve electrode contact |
| Firmware does not flash | Wrong COM port or monitor open | Close serial monitors and check Device Manager |
| Buzzer does not sound | GPIO mismatch or current issue | Confirm GPIO21 and use a driver transistor |

