"""
disturbance_detection.py
------------------------
Rule-based power quality disturbance detection.

Default thresholds follow IEC 61000-4-30 / EN 50160 conventions but are
fully configurable.  Users should be aware that different electrical
standards define thresholds differently; these defaults are indicative only.
"""

import numpy as np
import pandas as pd
from modules.signal_processing import rolling_rms


# ---------------------------------------------------------------------------
# Default threshold table
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    # Voltage ratio = V_measured / V_nominal
    "normal_low":         0.90,   # below this → sag (EN 50160: 0.90)
    "normal_high":        1.10,   # above this → swell
    "sag_threshold":      0.90,
    "swell_threshold":    1.10,
    "interruption_threshold": 0.10,  # < 10 % nominal → interruption
    # Frequency deviation (Hz from nominal)
    "freq_deviation_hz":  0.5,    # ±0.5 Hz  (EN 50160 interconnected)
    # THD
    "thd_warning_pct":    5.0,    # IEC/EN 50160: 8 % for LV, 5 % conservative
    "thd_high_pct":       8.0,
}


# ---------------------------------------------------------------------------
# Single-snapshot rule-based classification
# ---------------------------------------------------------------------------

def classify_snapshot(
    vrms: float,
    nominal_voltage: float,
    frequency: float,
    nominal_frequency: float,
    thd_pct: float | None,
    thresholds: dict | None = None,
) -> dict:
    """
    Classify a single electrical measurement snapshot.

    Returns
    -------
    dict with keys: status, disturbances (list of str), details (dict)
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    v_ratio = vrms / nominal_voltage if nominal_voltage > 0 else 1.0
    f_dev = abs(frequency - nominal_frequency)

    disturbances = []
    details: dict = {
        "voltage_ratio": round(v_ratio, 4),
        "frequency_deviation_hz": round(f_dev, 4),
    }

    # Voltage classification
    if v_ratio < thr["interruption_threshold"]:
        disturbances.append("Voltage Interruption")
        details["voltage_status"] = "Interruption"
    elif v_ratio < thr["sag_threshold"]:
        disturbances.append("Voltage Sag")
        details["voltage_status"] = "Sag"
    elif v_ratio > thr["swell_threshold"]:
        disturbances.append("Voltage Swell")
        details["voltage_status"] = "Swell"
    else:
        details["voltage_status"] = "Normal"

    # Frequency
    if f_dev > thr["freq_deviation_hz"]:
        disturbances.append("Frequency Deviation")
        details["frequency_status"] = f"Deviation {f_dev:+.3f} Hz"
    else:
        details["frequency_status"] = "Normal"

    # THD
    if thd_pct is not None:
        if thd_pct >= thr["thd_high_pct"]:
            disturbances.append("Harmonic Distortion (High)")
            details["thd_status"] = f"High THD {thd_pct:.1f}%"
        elif thd_pct >= thr["thd_warning_pct"]:
            disturbances.append("Harmonic Distortion (Elevated)")
            details["thd_status"] = f"Elevated THD {thd_pct:.1f}%"
        else:
            details["thd_status"] = f"Normal THD {thd_pct:.1f}%"
    else:
        details["thd_status"] = "THD not calculated"

    status = "Normal" if len(disturbances) == 0 else " | ".join(disturbances)
    return {"status": status, "disturbances": disturbances, "details": details}


# ---------------------------------------------------------------------------
# Time-series disturbance detection
# ---------------------------------------------------------------------------

def detect_disturbances_timeseries(
    df: pd.DataFrame,
    nominal_voltage: float = 230.0,
    nominal_frequency: float = 50.0,
    sampling_frequency: float = 3200.0,
    thresholds: dict | None = None,
    window_cycles: int = 1,
) -> pd.DataFrame:
    """
    Identify disturbance intervals in a time-series waveform.

    Uses a rolling RMS window of `window_cycles` cycles to track the envelope
    of the voltage waveform over time.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'timestamp' and 'voltage' columns.
    nominal_voltage : float
        Nominal RMS voltage (V).
    nominal_frequency : float
        Nominal frequency (Hz).
    sampling_frequency : float
        Samples per second.
    thresholds : dict or None
        Override specific threshold keys (see DEFAULT_THRESHOLDS).
    window_cycles : int
        Rolling-RMS window length in fundamental cycles.

    Returns
    -------
    pd.DataFrame with columns:
        disturbance_type, start_time, end_time, duration_s, severity, affected
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    voltage = df["voltage"].to_numpy(dtype=float)
    timestamps = df["timestamp"].to_numpy(dtype=float) if "timestamp" in df.columns else (
        np.arange(len(voltage)) / sampling_frequency
    )

    window_samples = max(1, int(window_cycles * sampling_frequency / nominal_frequency))
    v_rms_rolling = rolling_rms(voltage, window_samples)
    v_norm = v_rms_rolling / nominal_voltage

    # Build per-sample labels
    labels = np.full(len(v_norm), "normal", dtype=object)
    labels[v_norm < thr["interruption_threshold"]] = "interruption"
    labels[(v_norm >= thr["interruption_threshold"]) & (v_norm < thr["sag_threshold"])] = "voltage_sag"
    labels[v_norm > thr["swell_threshold"]] = "voltage_swell"

    events = []
    _collect_events(labels, timestamps, v_norm, nominal_voltage, events)

    if len(events) == 0:
        return pd.DataFrame(columns=[
            "disturbance_type", "start_time", "end_time",
            "duration_s", "severity", "affected"
        ])

    return pd.DataFrame(events)


def _collect_events(
    labels: np.ndarray,
    timestamps: np.ndarray,
    v_norm: np.ndarray,
    nominal_voltage: float,
    events: list,
):
    """Extract contiguous disturbance intervals from the label array."""
    n = len(labels)
    i = 0
    while i < n:
        lbl = labels[i]
        if lbl == "normal":
            i += 1
            continue
        # Find the end of this contiguous block
        j = i + 1
        while j < n and labels[j] == lbl:
            j += 1
        seg_vrms = v_norm[i:j]
        if lbl == "voltage_sag":
            severity = round(float(1.0 - np.min(seg_vrms)), 4)
        elif lbl == "voltage_swell":
            severity = round(float(np.max(seg_vrms) - 1.0), 4)
        elif lbl == "interruption":
            severity = round(float(1.0 - np.min(seg_vrms)), 4)
        else:
            severity = 0.0
        events.append({
            "disturbance_type": lbl.replace("_", " ").title(),
            "start_time":       round(float(timestamps[i]), 6),
            "end_time":         round(float(timestamps[j - 1]), 6),
            "duration_s":       round(float(timestamps[j - 1] - timestamps[i]), 6),
            "severity":         severity,
            "affected":         "Voltage",
        })
        i = j
