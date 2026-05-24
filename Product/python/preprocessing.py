"""
preprocessing.py
================
Signal preprocessing helpers shared between Python scripts.

Provides:
  - moving_average(signal, window_size)
  - baseline_remove(signal, window_size)
  - normalize_to_i16(value, adc_min, adc_max)
  - extract_features(window)

These functions mirror the logic implemented in the Rust firmware:
  firmware/esp32-rust/src/filter.rs
  firmware/esp32-rust/src/inference.rs

Usage:
    from preprocessing import moving_average, extract_features
"""

import numpy as np


# ─── Moving Average Filter ────────────────────────────────────────────────────

def moving_average(signal: np.ndarray, window_size: int = 8) -> np.ndarray:
    """
    Apply a causal moving average filter to the signal.

    This matches the Rust firmware's moving_average() in filter.rs.
    Uses a causal convolution (no future samples) to simulate real-time behavior.

    Args:
        signal     : 1D numpy array of ADC values
        window_size: Number of samples to average (default: 8)

    Returns:
        Filtered signal as float64 array, same length as input.
        Early samples (< window_size) are averaged over available samples.
    """
    filtered = np.zeros(len(signal), dtype=np.float64)
    for i in range(len(signal)):
        start = max(0, i - window_size + 1)
        filtered[i] = np.mean(signal[start : i + 1])
    return filtered


# ─── Baseline Removal ─────────────────────────────────────────────────────────

def baseline_remove(signal: np.ndarray, window_size: int = 64) -> np.ndarray:
    """
    Remove baseline wander using a long moving average as the baseline estimate.

    The baseline is estimated with a long window (e.g., 64 samples at 250 Hz = 256 ms).
    Subtracting the baseline isolates the AC (beat) component.

    Args:
        signal     : 1D numpy array of ADC values (float or int)
        window_size: Long window for baseline estimation (default: 64)

    Returns:
        Baseline-removed signal as float64. Values may be negative.
    """
    baseline = moving_average(signal, window_size)
    return signal.astype(np.float64) - baseline


# ─── ADC Normalization ────────────────────────────────────────────────────────

def normalize_adc_to_i16(value: float, adc_min: float = 0.0, adc_max: float = 4095.0) -> int:
    """
    Normalize a single ADC value from [adc_min, adc_max] → [-32768, 32767].

    This mirrors normalize_adc_to_i16() in firmware/esp32-rust/src/filter.rs.
    Used to prepare values for integer arithmetic in quantized inference.

    Args:
        value  : Raw ADC value
        adc_min: Minimum expected ADC value (default: 0)
        adc_max: Maximum expected ADC value (default: 4095)

    Returns:
        Normalized int16 value in range [-32768, 32767]
    """
    # Map to [-1.0, +1.0] first
    normalized_f32 = (value - adc_min) / (adc_max - adc_min) * 2.0 - 1.0
    # Scale to int16
    scaled = int(normalized_f32 * 32767.0)
    # Clamp to int16 range
    return max(-32768, min(32767, scaled))


def normalize_signal_to_i16(signal: np.ndarray, adc_min: float = 0.0, adc_max: float = 4095.0) -> np.ndarray:
    """
    Normalize an entire signal array to int16 range.
    Vectorized version of normalize_adc_to_i16().
    """
    normalized = (signal.astype(np.float64) - adc_min) / (adc_max - adc_min) * 2.0 - 1.0
    scaled = (normalized * 32767.0).astype(np.int32)
    return np.clip(scaled, -32768, 32767).astype(np.int16)


# ─── Feature Extraction ───────────────────────────────────────────────────────

def extract_features(window: np.ndarray) -> dict:
    """
    Extract a small set of features from an ECG window.

    These features are used as inputs to the quantized classifier.
    They mirror the feature extraction in firmware/esp32-rust/src/inference.rs.

    Features:
      - mean        : Average of the window (float)
      - maximum     : Maximum value in the window (float)
      - minimum     : Minimum value in the window (float)
      - peak_to_peak: max - min (float)
      - energy      : Sum of squared values (float)

    Args:
        window: 1D numpy array of filtered ADC values (window of samples)

    Returns:
        dict with keys: mean, maximum, minimum, peak_to_peak, energy
    """
    w = window.astype(np.float64)
    mean        = np.mean(w)
    maximum     = np.max(w)
    minimum     = np.min(w)
    peak_to_peak = maximum - minimum
    energy      = np.sum(w ** 2)

    return {
        "mean":         mean,
        "maximum":      maximum,
        "minimum":      minimum,
        "peak_to_peak": peak_to_peak,
        "energy":       energy,
    }


def extract_features_array(window: np.ndarray) -> np.ndarray:
    """
    Same as extract_features() but returns a numpy array instead of dict.
    Order: [mean, maximum, minimum, peak_to_peak, energy]
    Used for model training and inference.
    """
    feat = extract_features(window)
    return np.array([
        feat["mean"],
        feat["maximum"],
        feat["minimum"],
        feat["peak_to_peak"],
        feat["energy"],
    ], dtype=np.float64)


# ─── Window Generator ─────────────────────────────────────────────────────────

def sliding_windows(signal: np.ndarray, window_size: int = 128, step: int = 64):
    """
    Generate sliding windows over a signal for batch feature extraction.

    Args:
        signal     : 1D signal array
        window_size: Number of samples per window (default: 128 = 512 ms @ 250 Hz)
        step       : Step between windows (default: 64 = 50% overlap)

    Yields:
        (start_idx, window_array) tuples
    """
    n = len(signal)
    start = 0
    while start + window_size <= n:
        yield start, signal[start : start + window_size]
        start += step
