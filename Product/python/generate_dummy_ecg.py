"""
generate_dummy_ecg.py
=====================
Generates a synthetic ECG-like signal and saves it as:
    data/sample_ecg.csv

Signal structure:
  - 10 seconds at 250 Hz = 2500 samples
  - First 6 seconds: Normal ECG-like PQRST waveform
  - Next 4 seconds:  Abnormal ECG-like segment (elevated baseline,
                     irregular amplitude simulating arrhythmia)
  - Gaussian noise added throughout

ADC range: 0 – 4095 (12-bit, as on ESP32)
Baseline: ~2048 (midpoint, approximately 1.65V)

Usage:
    python python/generate_dummy_ecg.py

Output:
    data/sample_ecg.csv
"""

import os
import numpy as np
import csv

# ─── Configuration ────────────────────────────────────────────────────────────
SAMPLE_RATE_HZ  = 250          # Sampling frequency in Hz
DURATION_SEC    = 10           # Total signal duration in seconds
N_SAMPLES       = SAMPLE_RATE_HZ * DURATION_SEC   # 2500 samples
NORMAL_SEC      = 6            # First 6 seconds are "normal"
ADC_MIN         = 0
ADC_MAX         = 4095
ADC_MIDPOINT    = 2048         # Resting baseline (~1.65V)
HEART_RATE_BPM  = 72           # Simulated heart rate
NOISE_STD       = 30           # Gaussian noise standard deviation (ADC units)

OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "sample_ecg.csv")

# ─── ECG Waveform Builder ─────────────────────────────────────────────────────

def gaussian_peak(t_array, center, amplitude, width):
    """Returns a Gaussian-shaped peak centered at `center`."""
    return amplitude * np.exp(-0.5 * ((t_array - center) / width) ** 2)


def build_one_beat(t_relative, fs=250):
    """
    Build one synthetic ECG beat lasting 1/heart_rate seconds.
    Returns an array of shape (len(t_relative),) representing
    one PQRST complex centered in the beat period.

    PQRST approximate timing relative to beat center:
      P  wave: -0.25 s before R
      Q  wave: -0.05 s before R
      R  peak: 0.00 s (tallest peak)
      S  wave: +0.04 s after R
      T  wave: +0.20 s after R
    """
    beat_signal = np.zeros(len(t_relative))

    # P wave (small positive bump)
    beat_signal += gaussian_peak(t_relative, center=-0.25, amplitude=80,  width=0.05)
    # Q wave (small negative dip)
    beat_signal += gaussian_peak(t_relative, center=-0.05, amplitude=-60, width=0.02)
    # R peak (tall positive spike)
    beat_signal += gaussian_peak(t_relative, center=0.00,  amplitude=600, width=0.025)
    # S wave (small negative dip)
    beat_signal += gaussian_peak(t_relative, center=0.04,  amplitude=-80, width=0.02)
    # T wave (medium positive bump)
    beat_signal += gaussian_peak(t_relative, center=0.20,  amplitude=150, width=0.06)

    return beat_signal


def generate_normal_ecg(n_samples, fs=250, bpm=72):
    """
    Generate a normal ECG-like signal for n_samples at given sample rate.
    Returns array of ADC-like values centered around ADC_MIDPOINT.
    """
    t = np.arange(n_samples) / fs          # Time array in seconds
    beat_period = 60.0 / bpm               # Seconds per beat (e.g., ~0.833 s)

    signal = np.zeros(n_samples)

    # Iterate through each beat and superimpose onto signal
    beat_start = 0.0
    while beat_start < n_samples / fs:
        beat_center = beat_start + beat_period / 2.0
        # Relative time within the beat
        t_rel = t - beat_center
        # Mask to only the current beat window
        mask = (t_rel >= -beat_period / 2.0) & (t_rel < beat_period / 2.0)
        # Add the beat waveform
        signal[mask] += build_one_beat(t_rel[mask])
        beat_start += beat_period

    # Add Gaussian noise
    noise = np.random.normal(0, NOISE_STD, n_samples)
    signal += noise

    # Shift to ADC midpoint
    signal += ADC_MIDPOINT

    # Clip to valid ADC range
    signal = np.clip(signal, ADC_MIN, ADC_MAX).astype(int)
    return signal


def generate_abnormal_ecg(n_samples, fs=250, bpm=72):
    """
    Generate an abnormal ECG-like segment:
      - Elevated baseline (simulates ST elevation)
      - Irregular amplitude variation (simulates arrhythmia)
      - Wider P wave and irregular R-R interval
      - Larger noise
    """
    t = np.arange(n_samples) / fs
    beat_period = 60.0 / bpm

    signal = np.zeros(n_samples)

    beat_start = 0.0
    beat_index = 0
    while beat_start < n_samples / fs:
        # Irregular RR interval: ±20% variation
        variation = np.random.uniform(0.8, 1.2)
        current_period = beat_period * variation
        beat_center = beat_start + current_period / 2.0

        # Irregular amplitude: ±30% variation
        amp_factor = np.random.uniform(0.7, 1.4)

        t_rel = t - beat_center
        mask = (t_rel >= -current_period / 2.0) & (t_rel < current_period / 2.0)

        if mask.any():
            beat_waveform = build_one_beat(t_rel[mask])
            beat_waveform *= amp_factor
            signal[mask] += beat_waveform

        beat_start += current_period
        beat_index += 1

    # Elevated baseline (ST elevation simulation: ~+200 ADC units)
    baseline_elevation = 200
    # Add a slow drift to simulate baseline wander
    slow_drift = 100 * np.sin(2 * np.pi * 0.1 * t)

    # Larger noise
    noise = np.random.normal(0, NOISE_STD * 2, n_samples)

    signal += ADC_MIDPOINT + baseline_elevation + slow_drift + noise
    signal = np.clip(signal, ADC_MIN, ADC_MAX).astype(int)
    return signal


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("[generate_dummy_ecg] Generating synthetic ECG data...")
    print(f"  Sample rate : {SAMPLE_RATE_HZ} Hz")
    print(f"  Duration    : {DURATION_SEC} s")
    print(f"  Total samples: {N_SAMPLES}")
    print(f"  Normal segment: 0 – {NORMAL_SEC} s")
    print(f"  Abnormal segment: {NORMAL_SEC} – {DURATION_SEC} s")

    np.random.seed(42)  # Reproducible output

    # Normal segment: first NORMAL_SEC seconds
    n_normal   = SAMPLE_RATE_HZ * NORMAL_SEC
    n_abnormal = N_SAMPLES - n_normal

    normal_signal   = generate_normal_ecg(n_normal,   SAMPLE_RATE_HZ, HEART_RATE_BPM)
    abnormal_signal = generate_abnormal_ecg(n_abnormal, SAMPLE_RATE_HZ, HEART_RATE_BPM)

    # Concatenate
    all_adc    = np.concatenate([normal_signal, abnormal_signal])
    all_labels = np.concatenate([
        np.zeros(n_normal,   dtype=int),   # label 0 = Normal
        np.ones(n_abnormal,  dtype=int)    # label 1 = Abnormal
    ])

    # Timestamps in milliseconds (4 ms per sample @ 250 Hz)
    timestamps = np.arange(N_SAMPLES) * (1000 // SAMPLE_RATE_HZ)

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_ms", "adc_value", "label"])
        for ts, adc, lbl in zip(timestamps, all_adc, all_labels):
            writer.writerow([ts, adc, lbl])

    print(f"\n[generate_dummy_ecg] Saved {N_SAMPLES} samples to: {OUTPUT_CSV}")
    print(f"  Normal samples  : {n_normal}")
    print(f"  Abnormal samples: {n_abnormal}")
    print(f"  ADC range in output: {all_adc.min()} – {all_adc.max()}")
    print("[generate_dummy_ecg] Done.")


if __name__ == "__main__":
    main()
