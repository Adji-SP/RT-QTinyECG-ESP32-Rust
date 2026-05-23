# python/uart_feed_evaluator.py
# ==============================
# Real-time UART Feed Evaluator
#
# Sends simulated ECG data from PC to ESP32 via serial UART,
# captures the ESP32's int8 inference predictions, and simultaneously
# runs the PC's float32 sklearn model on the same data.
# Saves both prediction sets for comparison by compare_models.py.
#
# Usage:
#   py python/uart_feed_evaluator.py --port COM3 --baud 115200
#   py python/uart_feed_evaluator.py --port COM3 --dry-run   (no hardware)
#
# Protocol (ESP32 must be in UART_FEED_MODE):
#   PC  →  ESP32 : "2048\n"       (ADC value as ASCII integer)
#   ESP32 → PC   : "0\n" or "1\n" (prediction: 0=Normal, 1=Abnormal)
#                  "-1\n"          (buffer not yet full, no inference ran)
#
# DISCLAIMER: Educational prototype only. Not for clinical use.

import os
import sys
import time
import argparse
import csv
import pickle
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.join(SCRIPT_DIR, "..")
DATA_DIR    = os.path.join(ROOT_DIR, "data")
ECG_CSV     = os.path.join(DATA_DIR, "sample_ecg.csv")
MODEL_PKL   = os.path.join(DATA_DIR, "model.pkl")
SCALER_PKL  = os.path.join(DATA_DIR, "scaler.pkl")
OUT_CSV     = os.path.join(DATA_DIR, "esp32_predictions.csv")

# ── Signal processing constants (must match firmware) ─────────────────────────
RING_BUF_SIZE   = 128
FILTER_WINDOW   = 8
THRESHOLD       = 2800    # Firmware threshold for fallback classifier

# ── Default settings ──────────────────────────────────────────────────────────
DEFAULT_BAUD    = 115200
SAMPLE_INTERVAL = 1.0 / 250.0   # 4 ms between samples
UART_TIMEOUT    = 2.0            # seconds to wait for ESP32 response


def parse_args():
    p = argparse.ArgumentParser(
        description="RT-QTinyECG UART Feed Evaluator — PC→ESP32 data pipeline"
    )
    p.add_argument("--port",    default="COM3",         help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    p.add_argument("--baud",    default=DEFAULT_BAUD,   type=int, help="UART baud rate")
    p.add_argument("--dry-run", action="store_true",    help="Run without hardware (simulate ESP32 using PC)")
    p.add_argument("--data",    default=ECG_CSV,        help="Input ECG CSV file")
    p.add_argument("--delay",   default=0.0,            type=float, help="Extra delay between samples (ms)")
    return p.parse_args()


# ── Moving average filter (mirrors firmware filter.rs) ────────────────────────
class MovingAverage:
    def __init__(self, window=FILTER_WINDOW):
        self.window = window
        self.buf = [0] * window
        self.idx = 0
        self.sum = 0

    def push(self, val):
        self.sum -= self.buf[self.idx]
        self.buf[self.idx] = val
        self.sum += val
        self.idx = (self.idx + 1) % self.window
        return self.sum // self.window


# ── Threshold classifier (mirrors firmware inference.rs fallback) ──────────────
def threshold_infer(ring_buf):
    if len(ring_buf) < RING_BUF_SIZE:
        return -1  # buffer not full
    mean_val = sum(ring_buf) / len(ring_buf)
    return 1 if mean_val > THRESHOLD else 0


# ── Load PC sklearn model ─────────────────────────────────────────────────────
def load_pc_model():
    if os.path.exists(MODEL_PKL) and os.path.exists(SCALER_PKL):
        with open(MODEL_PKL, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PKL, "rb") as f:
            scaler = pickle.load(f)
        print(f"[uart_feed_evaluator] Loaded PC model: {MODEL_PKL}")
        return model, scaler
    else:
        print("[uart_feed_evaluator] WARNING: No trained model found. Using threshold classifier for PC.")
        return None, None


# ── Extract features from window (mirrors Python preprocessing.py) ────────────
def extract_features(window):
    arr = np.array(window, dtype=np.float32)
    mean    = float(np.mean(arr))
    std     = float(np.std(arr))
    mn      = float(np.min(arr))
    mx      = float(np.max(arr))
    ptp     = float(mx - mn)
    return np.array([[mean, std, mn, mx, ptp]])


# ── Dry-run: simulate ESP32 int8 model on PC ─────────────────────────────────
def simulate_esp32(ring_buf):
    """Simulate the ESP32 int8 threshold classifier. Replace with
    loaded quantized weights for a more realistic simulation."""
    return threshold_infer(ring_buf)


# ── Main evaluation loop ──────────────────────────────────────────────────────
def run_evaluation(args):
    # Load input data
    if not os.path.exists(args.data):
        print(f"[uart_feed_evaluator] ERROR: {args.data} not found. Run generate_dummy_ecg.py first.")
        sys.exit(1)

    df = pd.read_csv(args.data, comment="#")
    df.columns = df.columns.str.strip()

    # Try to find adc and label columns
    adc_col = None
    for c in ["adc_raw", "adc_value", "value"]:
        if c in df.columns:
            adc_col = c
            break
    if adc_col is None:
        adc_col = df.columns[1]  # fallback: second column

    label_col = None
    for c in ["label", "gt_label", "ground_truth"]:
        if c in df.columns:
            label_col = c
            break
    if label_col is None and len(df.columns) >= 3:
        label_col = df.columns[2]

    print(f"[uart_feed_evaluator] Loaded {len(df)} samples from {args.data}")
    print(f"[uart_feed_evaluator] ADC column: '{adc_col}', Label column: '{label_col}'")

    # Load PC model
    pc_model, pc_scaler = load_pc_model()

    # Open serial port (skip if dry-run)
    ser = None
    if not args.dry_run:
        try:
            import serial
            ser = serial.Serial(
                port=args.port,
                baudrate=args.baud,
                timeout=UART_TIMEOUT,
                write_timeout=UART_TIMEOUT
            )
            time.sleep(2.0)  # wait for ESP32 to boot
            ser.flushInput()
            ser.flushOutput()
            print(f"[uart_feed_evaluator] Serial opened: {args.port} @ {args.baud} baud")
        except Exception as e:
            print(f"[uart_feed_evaluator] ERROR opening serial port: {e}")
            print("  → Use --dry-run to test without hardware")
            sys.exit(1)
    else:
        print("[uart_feed_evaluator] DRY-RUN mode: simulating ESP32 on PC (no hardware)")

    # State
    filt         = MovingAverage(FILTER_WINDOW)
    ring_buf     = []
    pc_ring_buf  = []
    results      = []
    errors       = 0
    t_start      = time.perf_counter()

    print(f"\n[uart_feed_evaluator] Starting feed — {len(df)} samples @ 250 Hz")
    print("─" * 60)

    for idx, row in df.iterrows():
        adc_raw = int(row[adc_col])
        gt_label = int(row[label_col]) if label_col else -1

        # ── Apply moving average filter (same as firmware) ──────────────────
        filtered = filt.push(adc_raw)

        # ── Update ring buffer ───────────────────────────────────────────────
        if len(ring_buf) < RING_BUF_SIZE:
            ring_buf.append(filtered)
        else:
            ring_buf.pop(0)
            ring_buf.append(filtered)

        # ── PC float32 model prediction ──────────────────────────────────────
        if len(ring_buf) == RING_BUF_SIZE:
            if pc_model is not None:
                feats = extract_features(ring_buf)
                feats_scaled = pc_scaler.transform(feats)
                pc_pred = int(pc_model.predict(feats_scaled)[0])
            else:
                pc_pred = threshold_infer(ring_buf)
        else:
            pc_pred = -1

        # ── ESP32 prediction (via UART or simulation) ─────────────────────────
        t0 = time.perf_counter()

        if ser is not None:
            # Send ADC value to ESP32
            try:
                ser.write(f"{adc_raw}\n".encode("utf-8"))
                response = ser.readline().decode("utf-8").strip()
                esp32_pred = int(response) if response in ("0", "1", "-1") else -1
                round_trip_ms = (time.perf_counter() - t0) * 1000
            except Exception as e:
                esp32_pred = -1
                round_trip_ms = 0.0
                errors += 1
        else:
            # Dry-run: simulate ESP32
            esp32_pred = simulate_esp32(ring_buf)
            round_trip_ms = 0.0

        # ── Record result ────────────────────────────────────────────────────
        time_ms = int((time.perf_counter() - t_start) * 1000)
        results.append({
            "index":          idx,
            "time_ms":        time_ms,
            "adc_raw":        adc_raw,
            "filtered":       filtered,
            "gt_label":       gt_label,
            "pc_prediction":  pc_pred,
            "esp32_prediction": esp32_pred,
            "round_trip_ms":  round_trip_ms,
        })

        # ── Progress ─────────────────────────────────────────────────────────
        if idx % 250 == 0:
            progress = 100 * idx / len(df)
            print(f"  [{progress:5.1f}%] sample {idx:4d}  adc={adc_raw:4d}  "
                  f"pc={pc_pred:2d}  esp32={esp32_pred:2d}  gt={gt_label}")

        # ── Throttle to ~250 Hz if using real hardware ────────────────────────
        if ser is not None and args.delay > 0:
            time.sleep(args.delay / 1000.0)

    if ser:
        ser.close()

    # ── Save results ──────────────────────────────────────────────────────────
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)

    print("\n" + "─" * 60)
    print(f"[uart_feed_evaluator] Done.")
    print(f"  Samples processed : {len(results)}")
    print(f"  UART errors       : {errors}")
    if not args.dry_run and results:
        valid_rt = [r["round_trip_ms"] for r in results if r["round_trip_ms"] > 0]
        if valid_rt:
            print(f"  Mean round-trip   : {sum(valid_rt)/len(valid_rt):.1f} ms")
            print(f"  Max  round-trip   : {max(valid_rt):.1f} ms")
    print(f"  Output saved to   : {OUT_CSV}")
    print("─" * 60)

    return out_df


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
