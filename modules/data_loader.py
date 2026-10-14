"""
data_loader.py
--------------
Synthetic electrical data generation and CSV dataset loading / validation.
"""

import numpy as np
import pandas as pd
from io import StringIO


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    scenario: str = "normal",
    nominal_voltage: float = 230.0,
    nominal_frequency: float = 50.0,
    sampling_frequency: float = 3200.0,
    duration: float = 0.1,
    severity: float = 0.5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic single-phase voltage and current waveform.

    Parameters
    ----------
    scenario : str
        One of 'normal', 'voltage_sag', 'voltage_swell', 'interruption',
        'harmonic_distortion', 'frequency_deviation', 'multiple'.
    nominal_voltage : float
        Nominal RMS voltage (V).
    nominal_frequency : float
        Nominal fundamental frequency (Hz).
    sampling_frequency : float
        Samples per second (Hz).
    duration : float
        Waveform duration in seconds.
    severity : float
        0–1 scale controlling the magnitude of the injected disturbance.
    random_seed : int
        Reproducibility seed.

    Returns
    -------
    pd.DataFrame with columns: timestamp, voltage, current, frequency
    """
    rng = np.random.default_rng(random_seed)
    n_samples = int(sampling_frequency * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    v_peak = nominal_voltage * np.sqrt(2)
    i_peak = v_peak / 10.0          # assume 10-Ω nominal load
    freq = nominal_frequency        # fundamental frequency used in waveform

    # Build base waveform
    voltage = v_peak * np.sin(2 * np.pi * freq * t)
    current = i_peak * np.sin(2 * np.pi * freq * t - np.deg2rad(10))  # slight lag

    # ---- disturbance injection ----
    if scenario == "voltage_sag":
        # Reduce amplitude in central 40 % of the waveform
        start, end = int(0.3 * n_samples), int(0.7 * n_samples)
        sag_depth = 0.3 + 0.5 * severity          # 30–80 % reduction
        voltage[start:end] *= (1.0 - sag_depth)
        current[start:end] *= (1.0 - sag_depth)

    elif scenario == "voltage_swell":
        start, end = int(0.3 * n_samples), int(0.7 * n_samples)
        swell_mag = 0.1 + 0.4 * severity           # 10–50 % increase
        voltage[start:end] *= (1.0 + swell_mag)
        current[start:end] *= (1.0 + swell_mag)

    elif scenario == "interruption":
        start, end = int(0.2 * n_samples), int(0.8 * n_samples)
        voltage[start:end] *= (0.05 * (1.0 - severity))
        current[start:end] *= (0.05 * (1.0 - severity))

    elif scenario == "harmonic_distortion":
        thd_target = 0.05 + 0.35 * severity        # 5–40 %
        for h, coeff in [(3, 0.5), (5, 0.3), (7, 0.15), (11, 0.05)]:
            amp = v_peak * thd_target * coeff
            voltage += amp * np.sin(2 * np.pi * h * freq * t + rng.uniform(0, np.pi))
        # current harmonics (nonlinear load)
        for h, coeff in [(3, 0.5), (5, 0.3)]:
            current += (i_peak * thd_target * coeff) * np.sin(
                2 * np.pi * h * freq * t + rng.uniform(0, np.pi)
            )

    elif scenario == "frequency_deviation":
        dev = 2.5 * severity                        # up to ±2.5 Hz
        freq_dev = freq + dev
        voltage = v_peak * np.sin(2 * np.pi * freq_dev * t)
        current = i_peak * np.sin(2 * np.pi * freq_dev * t - np.deg2rad(10))

    elif scenario == "multiple":
        # Sag + harmonic
        start, end = int(0.3 * n_samples), int(0.7 * n_samples)
        voltage[start:end] *= 0.75
        current[start:end] *= 0.75
        for h, coeff in [(3, 0.4), (5, 0.2)]:
            voltage += (v_peak * 0.1 * coeff) * np.sin(2 * np.pi * h * freq * t)

    # Add small Gaussian noise
    voltage += rng.normal(0, v_peak * 0.005, n_samples)
    current += rng.normal(0, i_peak * 0.005, n_samples)

    # Instantaneous frequency column (constant or deviated)
    freq_col = np.full(n_samples, freq if scenario != "frequency_deviation" else freq + 2.5 * severity)

    df = pd.DataFrame({
        "timestamp": t,
        "voltage": voltage,
        "current": current,
        "frequency": freq_col,
    })
    return df


# ---------------------------------------------------------------------------
# CSV loading and validation
# ---------------------------------------------------------------------------

REQUIRED_COLS = {"voltage"}   # minimum required
OPTIONAL_COLS = {"timestamp", "current", "frequency"}
NUMERIC_COLS  = {"voltage", "current", "frequency"}


def load_csv(file_obj) -> tuple[pd.DataFrame | None, str]:
    """
    Load and validate a CSV file uploaded via Streamlit.

    Returns (dataframe, error_message).  error_message is empty on success.
    """
    try:
        df = pd.read_csv(file_obj)
    except Exception as exc:
        return None, f"Could not parse CSV file: {exc}"

    df.columns = df.columns.str.strip().str.lower()

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        return None, (
            f"CSV is missing required column(s): {', '.join(sorted(missing))}. "
            f"Expected at least: voltage."
        )

    # Cast numeric columns
    for col in NUMERIC_COLS & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where voltage is NaN
    before = len(df)
    df = df.dropna(subset=["voltage"])
    dropped = before - len(df)

    if len(df) == 0:
        return None, "The voltage column contains no valid numeric values."

    # Synthesise timestamp if missing
    if "timestamp" not in df.columns:
        df.insert(0, "timestamp", np.arange(len(df)) / 3200.0)

    if dropped > 0:
        msg = f"Warning: {dropped} row(s) with invalid voltage values were removed."
    else:
        msg = ""

    return df, msg


# ---------------------------------------------------------------------------
# Sample CSV content
# ---------------------------------------------------------------------------

def get_sample_csv() -> str:
    """Return a sample CSV string for download."""
    df = generate_synthetic_data(
        scenario="voltage_sag",
        nominal_voltage=230.0,
        nominal_frequency=50.0,
        sampling_frequency=3200.0,
        duration=0.1,
        severity=0.5,
    )
    # Keep every 4th sample to reduce file size (800 Hz effective)
    df = df.iloc[::4].reset_index(drop=True)
    return df.to_csv(index=False)
