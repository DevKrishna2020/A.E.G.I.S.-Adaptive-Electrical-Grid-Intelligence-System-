"""
signal_processing.py
--------------------
Signal-processing utilities: RMS, frequency estimation, FFT / THD analysis.
"""

import numpy as np
from scipy.signal import windows
from scipy.fft import rfft, rfftfreq


# ---------------------------------------------------------------------------
# RMS
# ---------------------------------------------------------------------------

def compute_rms(signal: np.ndarray) -> float:
    """Return the true RMS of a 1-D array."""
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2)))


# ---------------------------------------------------------------------------
# Fundamental frequency estimation
# ---------------------------------------------------------------------------

def estimate_frequency(signal: np.ndarray, sampling_rate: float, f_nominal: float = 50.0) -> float:
    """
    Estimate the fundamental frequency using the magnitude spectrum.

    A Hann window is applied before the FFT to reduce spectral leakage.
    The search is restricted to a ±20 % band around the nominal frequency to
    avoid identifying harmonic peaks instead of the fundamental.

    Parameters
    ----------
    signal : np.ndarray
        1-D array of voltage or current samples.
    sampling_rate : float
        Sampling frequency in Hz.
    f_nominal : float
        Expected fundamental frequency (Hz).  Used to constrain the search.

    Returns
    -------
    float : estimated fundamental frequency in Hz.
    """
    n = len(signal)
    if n < 2:
        return f_nominal

    win = windows.hann(n)
    spectrum = np.abs(rfft(signal * win))
    freqs = rfftfreq(n, d=1.0 / sampling_rate)

    # Restrict to ±20 % around nominal
    f_lo = f_nominal * 0.80
    f_hi = f_nominal * 1.20
    mask = (freqs >= f_lo) & (freqs <= f_hi)

    if not np.any(mask):
        return f_nominal

    idx_peak = np.argmax(spectrum[mask])
    return float(freqs[mask][idx_peak])


# ---------------------------------------------------------------------------
# THD and harmonic analysis
# ---------------------------------------------------------------------------

def compute_thd(signal: np.ndarray, sampling_rate: float, f1: float,
                max_harmonic: int = 20) -> tuple[float, dict]:
    """
    Compute voltage THD using the RMS amplitude at each harmonic.

    A Hann window is applied to reduce spectral leakage.

    Parameters
    ----------
    signal : np.ndarray
        Waveform samples.
    sampling_rate : float
        Sampling frequency in Hz.
    f1 : float
        Fundamental frequency in Hz.
    max_harmonic : int
        Number of harmonics to include (h = 2 … max_harmonic).

    Returns
    -------
    thd_percent : float
        THD in percent.
    harmonics : dict
        {harmonic_order: rms_amplitude} including h=1.

    Limitations
    -----------
    - Accuracy depends on the spectral resolution (sampling_rate / N).
      More samples → better frequency resolution.
    - Hann windowing reduces leakage but slightly broadens peaks.
    - Results are unreliable if the waveform duration is < 1 cycle.
    """
    n = len(signal)
    if n < 2:
        return 0.0, {}

    win = windows.hann(n)
    # Amplitude correction for Hann window (sum of window / n ≈ 0.5)
    scale = 2.0 / np.sum(win)
    spectrum = np.abs(rfft(signal * win)) * scale
    freqs = rfftfreq(n, d=1.0 / sampling_rate)

    def _rms_at_frequency(f_target: float, bw_frac: float = 0.3) -> float:
        """Return peak amplitude near f_target (within bw_frac * f1)."""
        bw = bw_frac * f1
        mask = (freqs >= f_target - bw) & (freqs <= f_target + bw)
        if not np.any(mask):
            return 0.0
        return float(np.max(spectrum[mask])) / np.sqrt(2)  # peak → RMS

    harmonics: dict = {}
    for h in range(1, max_harmonic + 1):
        harmonics[h] = _rms_at_frequency(h * f1)

    v1 = harmonics.get(1, 0.0)
    if v1 < 1e-9:
        return 0.0, harmonics

    harmonic_sum_sq = sum(harmonics[h] ** 2 for h in range(2, max_harmonic + 1))
    thd_percent = 100.0 * np.sqrt(harmonic_sum_sq) / v1
    return float(thd_percent), harmonics


# ---------------------------------------------------------------------------
# Rolling-window RMS
# ---------------------------------------------------------------------------

def rolling_rms(signal: np.ndarray, window_samples: int) -> np.ndarray:
    """
    Compute a causal rolling RMS over a 1-D signal.

    Uses cumulative sums for efficiency.

    Parameters
    ----------
    signal : np.ndarray
    window_samples : int
        Number of samples in the rolling window.

    Returns
    -------
    np.ndarray same length as signal.
    """
    if window_samples < 1:
        window_samples = 1
    sq = signal ** 2
    cumsum = np.cumsum(sq)
    cumsum = np.concatenate(([0.0], cumsum))
    rolling_sum = cumsum[window_samples:] - cumsum[:-window_samples]
    rms = np.sqrt(rolling_sum / window_samples)
    # Pad the beginning with the first valid value
    pad = np.full(window_samples - 1, rms[0] if len(rms) > 0 else 0.0)
    return np.concatenate((pad, rms))


# ---------------------------------------------------------------------------
# Sampling-rate check
# ---------------------------------------------------------------------------

def check_nyquist(sampling_rate: float, f_fundamental: float, max_harmonic: int = 20) -> str:
    """
    Return a warning string if the sampling rate may be insufficient
    to resolve the requested harmonic, or an empty string if it is adequate.
    """
    nyquist = sampling_rate / 2.0
    f_max_harmonic = max_harmonic * f_fundamental
    if nyquist < f_max_harmonic:
        resolvable = int(nyquist // f_fundamental)
        return (
            f"Sampling rate {sampling_rate:.0f} Hz can resolve harmonics up to "
            f"h={resolvable} (Nyquist = {nyquist:.0f} Hz). "
            f"THD accuracy is limited to {resolvable} harmonics."
        )
    return ""
