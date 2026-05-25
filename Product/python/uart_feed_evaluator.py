# python/uart_feed_evaluator.py
# ==============================
# Real-time UART Feed Evaluator
#
# Sends simulated ECG data from PC to ESP32-S3 via serial UART,
# captures the ESP32-S3's int8 inference predictions, and simultaneously
# runs the PC's float32 sklearn model on the same data.
# Saves both prediction sets for comparison by compare_models.py.
#
# Usage:
#   py python/uart_feed_evaluator.py --port COM16 --baud 115200
#   py python/uart_feed_evaluator.py --port COM16 --dry-run
#
# Protocol, ESP32-S3 must be in UART_FEED_MODE:
#   PC  → ESP32-S3 : "2048\n"       ADC value as ASCII integer
#   ESP32-S3 → PC  : "0\n" or "1\n" prediction, 0=Normal, 1=Abnormal
#                    "-1\n"         buffer not yet full, no inference ran
#
# DISCLAIMER: Educational prototype only. Not for clinical use.

import os
import sys
import time
import argparse
import pickle
import numpy as np
import pandas as pd


# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(ROOT_DIR, "data")

sys.path.insert(0, SCRIPT_DIR)
from preprocessing import extract_firmware_mlp_input_array

ECG_CSV = os.path.join(DATA_DIR, "sample_ecg.csv")
MODEL_PKL = os.path.join(DATA_DIR, "model.pkl")
SCALER_PKL = os.path.join(DATA_DIR, "scaler.pkl")
WEIGHTS_NPZ = os.path.join(DATA_DIR, "quantized_weights.npz")
OUT_CSV = os.path.join(DATA_DIR, "esp32_predictions.csv")


# ── Signal processing constants, must match firmware ──────────────────────────
RING_BUF_SIZE = 128
FILTER_WINDOW = 8
THRESHOLD = 2800


# ── Default settings ──────────────────────────────────────────────────────────
DEFAULT_BAUD = 115200
DEFAULT_PORT = "COM16"

# Shorter timeout prevents the script from looking stuck if ESP32-S3 does not
# reply cleanly for a sample.
UART_TIMEOUT = 0.20

# 250 Hz = 4 ms per sample.
SAMPLE_INTERVAL = 1.0 / 250.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="RT-QTinyECG UART Feed Evaluator — PC to ESP32-S3 pipeline"
    )

    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help="Serial port, for example COM16 or /dev/ttyUSB0",
    )

    parser.add_argument(
        "--baud",
        default=DEFAULT_BAUD,
        type=int,
        help="UART baud rate",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without hardware. Simulate ESP32-S3 using PC.",
    )

    parser.add_argument(
        "--data",
        default=ECG_CSV,
        help="Input ECG CSV file",
    )

    parser.add_argument(
        "--delay",
        default=0.0,
        type=float,
        help="Extra delay between samples in milliseconds. Default: 0.0",
    )

    parser.add_argument(
        "--progress-every",
        default=25,
        type=int,
        help="Print progress every N samples. Default: 25",
    )

    return parser.parse_args()


# ── Moving average filter, mirrors firmware filter.rs ─────────────────────────
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


# ── Threshold classifier, mirrors firmware inference.rs fallback ──────────────
def threshold_infer(ring_buf):
    if len(ring_buf) < RING_BUF_SIZE:
        return -1

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

    print("[uart_feed_evaluator] WARNING: No trained model found.")
    print("[uart_feed_evaluator] Using threshold classifier for PC.")
    return None, None


# ── Extract features from window, mirrors preprocessing.py ────────────────────
def extract_features(window):
    return extract_firmware_mlp_input_array(np.array(window, dtype=np.int32)).reshape(1, -1)


def load_quantized_model():
    if not os.path.exists(WEIGHTS_NPZ):
        print("[uart_feed_evaluator] WARNING: No quantized weights found for dry-run ESP32 simulation.")
        return None

    data = np.load(WEIGHTS_NPZ)
    w1 = data["W_q_0"].astype(np.int32)
    b1 = data["b_q_0"].astype(np.int32)
    w2 = data["W_q_1"].astype(np.int32)
    b2 = data["b_q_1"].astype(np.int32)

    # sklearn stores W1 as [features, hidden] and W2 as [hidden, output].
    # Firmware stores W1 as [hidden, features] and W2 as [hidden] for one output.
    if w1.shape[0] == 5:
        w1 = w1.T
    if len(w2.shape) == 2:
        w2 = w2.reshape(-1)

    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


# ── Dry-run: simulate ESP32-S3 int8 model on PC ───────────────────────────────
def simulate_esp32(ring_buf, qmodel=None):
    if len(ring_buf) < RING_BUF_SIZE:
        return -1
    if qmodel is None:
        return threshold_infer(ring_buf)

    feat_q = extract_features(ring_buf).reshape(-1).astype(np.int32)
    hidden = np.maximum(qmodel["w1"].dot(feat_q) + qmodel["b1"], 0)
    h_max = max(1, int(np.max(hidden)))
    hidden_q = np.clip((hidden * 127) // h_max, -128, 127).astype(np.int32)
    out = int(qmodel["w2"].dot(hidden_q) + qmodel["b2"][0])
    return 1 if out > 0 else 0


# ── Serial helpers ────────────────────────────────────────────────────────────
def read_esp32_prediction(ser, max_lines=8):
    """
    Read ESP32-S3 response and ignore boot logs or debug text.

    Valid protocol response must be:
      -1, 0, or 1

    Returns:
      tuple[int, str]: prediction and raw valid response text.
    """
    for _ in range(max_lines):
        raw = ser.readline()

        if not raw:
            continue

        text = raw.decode("utf-8", errors="ignore").strip()

        if text in ("-1", "0", "1"):
            return int(text), text

        # Ignore ESP-IDF boot logs / startup banner / debug lines.
        # Examples:
        #   I (29) boot: ESP-IDF ...
        #   # UART_FEED_MODE ready...
        continue

    return -1, ""


def open_serial_port(port, baud):
    try:
        import serial

        ser = serial.Serial(
            port=port,
            baudrate=baud,
            timeout=UART_TIMEOUT,
            write_timeout=UART_TIMEOUT,
            rtscts=False,
            dsrdtr=False,
        )

        # Avoid unwanted reset toggling on some USB-UART adapters.
        # Some ESP boards still reset once when the port opens; that is normal.
        ser.dtr = False
        ser.rts = False

        print(f"[uart_feed_evaluator] Serial opened: {port} @ {baud} baud")
        print("[uart_feed_evaluator] Waiting for ESP32-S3 boot/startup text...")

        time.sleep(2.0)

        # Drain ESP-IDF boot messages and startup banner.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            line = ser.readline()

            if not line:
                break

            text = line.decode("utf-8", errors="ignore").strip()

            if text:
                print(f"[boot] {text}")

        ser.reset_input_buffer()
        ser.reset_output_buffer()

        print("[uart_feed_evaluator] Serial ready.")
        return ser

    except Exception as e:
        print(f"[uart_feed_evaluator] ERROR opening serial port: {e}")
        print("  → Make sure espflash monitor / Arduino monitor / PuTTY is closed.")
        print("  → Or use --dry-run to test without hardware.")
        sys.exit(1)


# ── Main evaluation loop ──────────────────────────────────────────────────────
def run_evaluation(args):
    # ── Load input data ───────────────────────────────────────────────────────
    if not os.path.exists(args.data):
        print(f"[uart_feed_evaluator] ERROR: {args.data} not found.")
        print("Run generate_dummy_ecg.py first, or pass another file with --data.")
        sys.exit(1)

    df = pd.read_csv(args.data, comment="#")
    df.columns = df.columns.str.strip()

    # Try to find ADC and label columns.
    adc_col = None
    for c in ["adc_raw", "adc_value", "value"]:
        if c in df.columns:
            adc_col = c
            break

    if adc_col is None:
        adc_col = df.columns[1]

    label_col = None
    for c in ["label", "gt_label", "ground_truth"]:
        if c in df.columns:
            label_col = c
            break

    if label_col is None and len(df.columns) >= 3:
        label_col = df.columns[2]

    print(f"[uart_feed_evaluator] Loaded {len(df)} samples from {args.data}")
    print(f"[uart_feed_evaluator] ADC column: '{adc_col}', Label column: '{label_col}'")

    # ── Load PC model ─────────────────────────────────────────────────────────
    pc_model, pc_scaler = load_pc_model()
    qmodel = None

    # ── Open serial port, skip if dry-run ─────────────────────────────────────
    ser = None

    if not args.dry_run:
        ser = open_serial_port(args.port, args.baud)
    else:
        print("[uart_feed_evaluator] DRY-RUN mode: simulating ESP32-S3 on PC.")
        qmodel = load_quantized_model()

    # ── State ─────────────────────────────────────────────────────────────────
    filt = MovingAverage(FILTER_WINDOW)
    ring_buf = []
    results = []
    errors = 0
    timeout_count = 0
    invalid_count = 0

    t_start = time.perf_counter()

    print()
    print(f"[uart_feed_evaluator] Starting feed — {len(df)} samples @ 250 Hz")
    print("─" * 60)

    for idx, row in df.iterrows():
        adc_raw = int(row[adc_col])
        gt_label = int(row[label_col]) if label_col else -1

        # ── Apply moving average filter, same as firmware ────────────────────
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
                model_input = pc_scaler.transform(feats) if pc_scaler is not None else feats
                pc_pred = int(pc_model.predict(model_input)[0])
            else:
                pc_pred = threshold_infer(ring_buf)
        else:
            pc_pred = -1

        # ── ESP32-S3 prediction, via UART or simulation ──────────────────────
        t0 = time.perf_counter()

        if ser is not None:
            try:
                ser.write(f"{adc_raw}\n".encode("utf-8"))
                ser.flush()

                esp32_pred, response = read_esp32_prediction(ser)
                round_trip_ms = (time.perf_counter() - t0) * 1000.0

                if response == "":
                    timeout_count += 1
                    errors += 1

                elif esp32_pred not in (-1, 0, 1):
                    invalid_count += 1
                    errors += 1
                    esp32_pred = -1

            except Exception as e:
                esp32_pred = -1
                round_trip_ms = 0.0
                errors += 1
        else:
            esp32_pred = simulate_esp32(ring_buf, qmodel)
            round_trip_ms = 0.0

        # ── Record result ────────────────────────────────────────────────────
        time_ms = int((time.perf_counter() - t_start) * 1000)

        results.append({
            "index": idx,
            "time_ms": time_ms,
            "adc_raw": adc_raw,
            "filtered": filtered,
            "gt_label": gt_label,
            "pc_prediction": pc_pred,
            "esp32_prediction": esp32_pred,
            "round_trip_ms": round_trip_ms,
        })

        # ── Progress ─────────────────────────────────────────────────────────
        if args.progress_every > 0 and idx % args.progress_every == 0:
            progress = 100.0 * idx / len(df)
            print(
                f"  [{progress:5.1f}%] sample {idx:4d}  "
                f"adc={adc_raw:4d}  "
                f"pc={pc_pred:2d}  "
                f"esp32={esp32_pred:2d}  "
                f"gt={gt_label}  "
                f"rt={round_trip_ms:6.1f} ms"
            )

        # ── Optional throttle ────────────────────────────────────────────────
        # Use --delay 4 if you want to force approximately 250 Hz from PC side.
        if ser is not None and args.delay > 0:
            time.sleep(args.delay / 1000.0)

    # ── Close serial ─────────────────────────────────────────────────────────
    if ser:
        ser.close()

    # ── Save results ─────────────────────────────────────────────────────────
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)

    print()
    print("─" * 60)
    print("[uart_feed_evaluator] Done.")
    print(f"  Samples processed : {len(results)}")
    print(f"  UART errors       : {errors}")
    print(f"  UART timeouts     : {timeout_count}")
    print(f"  Invalid responses : {invalid_count}")

    if not args.dry_run and results:
        valid_rt = [
            r["round_trip_ms"]
            for r in results
            if r["round_trip_ms"] > 0
        ]

        if valid_rt:
            print(f"  Mean round-trip   : {sum(valid_rt) / len(valid_rt):.1f} ms")
            print(f"  Max  round-trip   : {max(valid_rt):.1f} ms")

    print(f"  Output saved to   : {OUT_CSV}")
    print("─" * 60)

    return out_df


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
