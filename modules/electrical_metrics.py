"""
electrical_metrics.py
---------------------
Compute electrical parameters from raw waveform data or summary statistics.
"""

import numpy as np
import pandas as pd
from modules.signal_processing import (
    compute_rms,
    estimate_frequency,
    compute_thd,
    check_nyquist,
)


# ---------------------------------------------------------------------------
# Waveform-based full analysis
# ---------------------------------------------------------------------------

def compute_metrics_from_waveform(
    df: pd.DataFrame,
    nominal_voltage: float = 230.0,
    nominal_frequency: float = 50.0,
    sampling_frequency: float | None = None,
) -> dict:
    """
    Compute electrical metrics from a raw waveform DataFrame.

    Expected columns: timestamp, voltage, [current], [frequency]

    Returns
    -------
    dict with keys:
        vrms, irms, frequency, power_factor, thd_pct,
        active_power, apparent_power, reactive_power,
        thd_harmonics, nyquist_warning, has_current, source
    """
    result: dict = {"source": "waveform"}

    voltage = df["voltage"].to_numpy(dtype=float)
    has_current = "current" in df.columns and df["current"].notna().any()

    # Sampling rate
    if sampling_frequency is None:
        if "timestamp" in df.columns and len(df) > 1:
            dt = float(df["timestamp"].iloc[1] - df["timestamp"].iloc[0])
            fs = 1.0 / dt if dt > 0 else 3200.0
        else:
            fs = 3200.0
    else:
        fs = sampling_frequency

    # RMS
    vrms = compute_rms(voltage)
    result["vrms"] = round(vrms, 4)

    if has_current:
        current = df["current"].to_numpy(dtype=float)
        irms = compute_rms(current)
        result["irms"] = round(irms, 4)
        result["has_current"] = True
    else:
        current = None
        result["irms"] = None
        result["has_current"] = False

    # Frequency
    f1 = estimate_frequency(voltage, fs, nominal_frequency)
    result["frequency"] = round(f1, 3)

    # THD
    nyquist_warning = check_nyquist(fs, f1)
    result["nyquist_warning"] = nyquist_warning

    min_duration_cycles = 2
    n_min = int(min_duration_cycles / f1 * fs)
    if len(voltage) >= n_min:
        thd_pct, harmonics = compute_thd(voltage, fs, f1)
        result["thd_pct"] = round(thd_pct, 3)
        result["thd_harmonics"] = {k: round(v, 4) for k, v in harmonics.items()}
        result["thd_valid"] = True
    else:
        result["thd_pct"] = None
        result["thd_harmonics"] = {}
        result["thd_valid"] = False
        result["thd_note"] = "Insufficient waveform data for reliable THD calculation."

    # Power quantities
    if has_current:
        p = float(np.mean(voltage * current))         # active power W
        s = vrms * irms                                # apparent power VA
        pf = p / s if s > 1e-9 else 0.0               # power factor
        # Reactive power (valid only for sinusoidal, single-phase)
        q_sq = max(s ** 2 - p ** 2, 0.0)
        q = float(np.sqrt(q_sq))
        result["active_power"]   = round(p, 3)
        result["apparent_power"] = round(s, 3)
        result["reactive_power"] = round(q, 3)
        result["power_factor"]   = round(pf, 4)
        result["power_note"] = (
            "Reactive power is calculated as Q = √(S²−P²), which is valid only "
            "for sinusoidal, single-phase, steady-state conditions."
        )
    else:
        result["active_power"]   = None
        result["apparent_power"] = None
        result["reactive_power"] = None
        result["power_factor"]   = None

    return result


# ---------------------------------------------------------------------------
# Manual-input analysis
# ---------------------------------------------------------------------------

def compute_metrics_from_manual(
    vrms: float,
    irms: float,
    frequency: float,
    power_factor: float,
    thd_pct: float,
) -> dict:
    """
    Derive power metrics from user-supplied summary values.

    Note: THD and frequency values are taken directly from user input.
    No waveform analysis is performed.
    """
    s = vrms * irms
    p = s * power_factor
    q_sq = max(s ** 2 - p ** 2, 0.0)
    q = float(np.sqrt(q_sq))

    return {
        "source": "manual",
        "vrms": round(vrms, 4),
        "irms": round(irms, 4),
        "frequency": round(frequency, 3),
        "power_factor": round(power_factor, 4),
        "thd_pct": round(thd_pct, 3),
        "active_power": round(p, 3),
        "apparent_power": round(s, 3),
        "reactive_power": round(q, 3),
        "has_current": True,
        "thd_valid": True,
        "thd_note": "THD value was manually supplied by the user.",
        "nyquist_warning": "",
        "thd_harmonics": {},
    }


# ---------------------------------------------------------------------------
# Build a summary dict suitable for display
# ---------------------------------------------------------------------------

def metrics_summary(metrics: dict, nominal_voltage: float = 230.0) -> dict:
    """
    Return a flat dict of labelled strings for dashboard metric cards.
    """
    vrms = metrics.get("vrms")
    irms = metrics.get("irms")
    freq = metrics.get("frequency")
    pf   = metrics.get("power_factor")
    thd  = metrics.get("thd_pct")

    v_dev = None
    if vrms is not None:
        v_dev = 100.0 * (vrms - nominal_voltage) / nominal_voltage

    return {
        "RMS Voltage (V)":        f"{vrms:.2f}" if vrms is not None else "N/A",
        "Voltage Deviation (%)":  f"{v_dev:+.2f}" if v_dev is not None else "N/A",
        "RMS Current (A)":        f"{irms:.3f}" if irms is not None else "N/A",
        "Frequency (Hz)":         f"{freq:.3f}" if freq is not None else "N/A",
        "Power Factor":           f"{pf:.3f}" if pf is not None else "N/A",
        "Voltage THD (%)":        f"{thd:.2f}" if thd is not None else "Insufficient data",
    }
