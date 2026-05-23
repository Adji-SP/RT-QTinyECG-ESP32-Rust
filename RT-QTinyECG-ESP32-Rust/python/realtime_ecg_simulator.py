"""
realtime_ecg_simulator.py
==========================
Simulates the ESP32 real-time ECG pipeline on PC.

Loads data/sample_ecg.csv, runs through the same pipeline that the
ESP32 firmware runs (ring buffer → filter → inference → alert logic),
and saves a full CSV log to data/simulated_realtime_log.csv.

This allows the GNUPlot scripts to visualize the pipeline output
without needing real ESP32 hardware.

Pipeline:
  1. Load sample_ecg.csv  (ADC values at 250 Hz)
  2. Feed samples one at a time into a ring buffer (window_size=128)
  3. When buffer is full, apply moving average and extract features
  4. Run threshold-based or MLP inference
  5. Measure simulated inference time
  6. Activate alert if abnormal, measure alert latency
  7. Log each sample to CSV

Usage:
    python python/realtime_ecg_simulator.py

Output:
    data/simulated_realtime_log.csv
"""

import os
import csv
import time
import collections
import numpy as np

# Import our preprocessing helpers
import sys
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import moving_average, extract_features_array

# ─── Configuration ────────────────────────────────────────────────────────────
SAMPLE_RATE_HZ    = 250        # Simulated sampling rate
SAMPLE_INTERVAL_MS = 1000.0 / SAMPLE_RATE_HZ   # 4.0 ms
WINDOW_SIZE        = 128       # Samples per inference window
MOVING_AVG_SIZE    = 8         # Samples for moving average filter

# Threshold classifier parameters
# These mirror the threshold values in firmware/esp32-rust/src/inference.rs
THRESHOLD_PEAK_TO_PEAK = 600   # ADC units; above this = abnormal
THRESHOLD_MEAN_HIGH    = 2350  # Mean above this = elevated baseline (abnormal)
THRESHOLD_MEAN_LOW     = 1750  # Mean below this = depressed baseline (abnormal)

INPUT_CSV  = os.path.join(os.path.dirname(__file__), "..", "data", "sample_ecg.csv")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "simulated_realtime_log.csv")


# ─── Inference Functions ──────────────────────────────────────────────────────

def threshold_inference(window: np.ndarray) -> int:
    """
    Simple threshold-based abnormality detection.

    This directly mirrors the Rust firmware's threshold_classify()
    in firmware/esp32-rust/src/inference.rs.

    Returns:
        0 = Normal
        1 = Abnormal
    """
    w = window.astype(np.float64)
    mean         = np.mean(w)
    peak_to_peak = np.max(w) - np.min(w)

    # Rule 1: High peak-to-peak amplitude (erratic beats)
    if peak_to_peak > THRESHOLD_PEAK_TO_PEAK:
        return 1  # Abnormal

    # Rule 2: Elevated mean (ST elevation simulation)
    if mean > THRESHOLD_MEAN_HIGH:
        return 1  # Abnormal

    # Rule 3: Depressed mean (unusual baseline drop)
    if mean < THRESHOLD_MEAN_LOW:
        return 1  # Abnormal

    return 0  # Normal


def load_mlp_weights():
    """
    Load quantized MLP weights if available, else return None.

    Tries to import the weights exported by export_rust_weights.py.
    If not available, falls back to threshold inference only.
    """
    try:
        weights_file = os.path.join(
            os.path.dirname(__file__), "..", "firmware", "esp32-rust", "src", "model_weights.py"
        )
        if os.path.exists(weights_file):
            # This would import the Python-format weights
            pass
    except Exception:
        pass
    return None


def mlp_inference_manual(window: np.ndarray,
                          w1: np.ndarray, b1: np.ndarray,
                          w2: np.ndarray, b2: np.ndarray,
                          scale: float = 1.0 / 4096.0) -> int:
    """
    Manual integer-approximated MLP inference.

    This mirrors the Rust firmware's mlp_infer() in inference.rs.

    Architecture: 5 inputs → 8 hidden (ReLU) → 1 output (sigmoid)
    All computation in integer arithmetic for ESP32 simulation.

    Args:
        window : Signal window array
        w1, b1 : Layer 1 int8 weights and biases
        w2, b2 : Layer 2 int8 weights and biases
        scale  : Dequantization scale factor

    Returns:
        0 = Normal, 1 = Abnormal
    """
    features = extract_features_array(window)

    # Quantize input features to int8 range
    feat_max = max(abs(features.max()), abs(features.min()), 1e-6)
    feat_q = np.clip((features / feat_max * 127).astype(np.int32), -128, 127)

    # Layer 1: hidden = ReLU(W1 @ x + b1)  in int32 arithmetic
    hidden = np.dot(w1.astype(np.int32), feat_q.astype(np.int32)) + b1.astype(np.int32)
    hidden = np.maximum(hidden, 0)  # ReLU
    # Re-quantize hidden to int8
    h_max = max(abs(hidden.max()), 1)
    hidden_q = np.clip((hidden * 127 // h_max).astype(np.int32), -128, 127)

    # Layer 2: output = W2 @ hidden + b2
    output_q = int(np.dot(w2.astype(np.int32), hidden_q.astype(np.int32))) + int(b2[0])

    # Positive output → abnormal (sigmoid > 0.5 equivalent)
    return 1 if output_q > 0 else 0


# ─── Alert Logic ──────────────────────────────────────────────────────────────

class AlertStateMachine:
    """
    Tracks alert state and measures alert latency.

    Alert latency = time from abnormal detection to alert ON (ms).
    In real firmware, this is the time from inference completing
    to the GPIO toggle actually happening.
    On PC simulator, we simulate this as a small fixed delay.
    """
    def __init__(self, simulated_gpio_delay_ms: float = 1.5):
        self.alert_on        = False
        self.last_abnormal_ts: float | None = None
        self.alert_latency_ms = 0.0
        self.gpio_delay_ms   = simulated_gpio_delay_ms

    def update(self, prediction: int, current_time_ms: float) -> tuple[int, float]:
        """
        Update alert state.

        Returns:
            (alert_state: 0 or 1, alert_latency_ms: float)
        """
        if prediction == 1:
            if not self.alert_on:
                # First detection of abnormal
                self.alert_on = True
                self.alert_latency_ms = self.gpio_delay_ms  # GPIO toggle latency
            else:
                self.alert_latency_ms = 0.0
        else:
            if self.alert_on:
                self.alert_on = False
            self.alert_latency_ms = 0.0

        return (1 if self.alert_on else 0, self.alert_latency_ms)


# ─── Circular Ring Buffer ─────────────────────────────────────────────────────

class RingBuffer:
    """
    Fixed-size circular buffer for ECG samples.
    Mirrors RingBuffer<T, N> in firmware/esp32-rust/src/ring_buffer.rs.
    """
    def __init__(self, size: int):
        self.size   = size
        self.buffer = collections.deque(maxlen=size)

    def push(self, value: int):
        self.buffer.append(value)

    def is_full(self) -> bool:
        return len(self.buffer) == self.size

    def as_array(self) -> np.ndarray:
        return np.array(self.buffer, dtype=np.int32)

    def mean(self) -> float:
        if len(self.buffer) == 0:
            return 0.0
        return sum(self.buffer) / len(self.buffer)


# ─── Main Simulation Loop ─────────────────────────────────────────────────────

def main():
    print("[realtime_ecg_simulator] Loading ECG data...")

    # Load sample data
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: {INPUT_CSV} not found. Run generate_dummy_ecg.py first.")
        return

    timestamps   = []
    adc_values   = []
    ground_truth = []

    with open(INPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(int(row["time_ms"]))
            adc_values.append(int(row["adc_value"]))
            ground_truth.append(int(row["label"]))

    n_samples = len(adc_values)
    print(f"  Loaded {n_samples} samples from {INPUT_CSV}")
    print(f"[realtime_ecg_simulator] Starting simulation at {SAMPLE_RATE_HZ} Hz...")

    # Initialize components
    ring_buf   = RingBuffer(WINDOW_SIZE)
    alert_sm   = AlertStateMachine(simulated_gpio_delay_ms=1.5)
    log_rows   = []

    # Moving average state (causal, per-sample)
    recent_samples = collections.deque(maxlen=MOVING_AVG_SIZE)

    # Per-sample processing loop
    for i in range(n_samples):
        ts_ms = timestamps[i]
        adc   = adc_values[i]

        # ── Step 1: Push into moving average window ──
        recent_samples.append(adc)
        filtered_value = sum(recent_samples) / len(recent_samples)

        # ── Step 2: Push filtered value into ring buffer ──
        ring_buf.push(int(filtered_value))

        # ── Step 3: Run inference when buffer is full ──
        prediction    = 0
        inference_us  = 0.0
        alert         = 0
        latency_ms    = 0.0

        if ring_buf.is_full():
            window = ring_buf.as_array()

            # Measure inference time on PC
            t_start = time.perf_counter()
            prediction = threshold_inference(window)
            t_end   = time.perf_counter()

            inference_us = (t_end - t_start) * 1e6  # Convert to microseconds

            # ── Step 4: Update alert state machine ──
            alert, latency_ms = alert_sm.update(prediction, ts_ms)

        # ── Step 5: Log row ──
        log_rows.append({
            "time_ms":         ts_ms,
            "adc_value":       adc,
            "filtered_value":  round(filtered_value, 2),
            "inference_us":    round(inference_us, 2),
            "prediction":      prediction,
            "alert":           alert,
            "alert_latency_ms": round(latency_ms, 2),
        })

    # Write output CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = ["time_ms", "adc_value", "filtered_value",
                  "inference_us", "prediction", "alert", "alert_latency_ms"]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    # Summary statistics
    alerts_fired   = sum(1 for r in log_rows if r["alert"] == 1)
    predictions_ab = sum(1 for r in log_rows if r["prediction"] == 1)
    inf_times      = [r["inference_us"] for r in log_rows if r["inference_us"] > 0]

    print(f"\n[realtime_ecg_simulator] Simulation complete.")
    print(f"  Samples processed   : {n_samples}")
    print(f"  Abnormal predictions: {predictions_ab}")
    print(f"  Alert samples       : {alerts_fired}")
    if inf_times:
        print(f"  Avg inference time  : {np.mean(inf_times):.2f} µs")
        print(f"  Max inference time  : {np.max(inf_times):.2f} µs")
    print(f"  Output saved to     : {OUTPUT_CSV}")
    print("[realtime_ecg_simulator] Done.")


if __name__ == "__main__":
    main()
