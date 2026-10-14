"""
test_electrical_metrics.py
---------------------------
Automated tests for electrical parameter calculations.
Run with: python -m pytest tests/ -v
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from modules.signal_processing import compute_rms, estimate_frequency, compute_thd, rolling_rms
from modules.electrical_metrics import compute_metrics_from_waveform, compute_metrics_from_manual


# ── Helpers ────────────────────────────────────────────────────────────────

FS = 3200.0          # Hz
F1 = 50.0            # Hz
T  = 2.0             # seconds (long enough for good frequency resolution)
N  = int(FS * T)
t  = np.linspace(0, T, N, endpoint=False)


def pure_sine(amplitude: float, frequency: float = F1, phase: float = 0.0) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


# ── RMS tests ──────────────────────────────────────────────────────────────

class TestComputeRMS:
    def test_sine_rms(self):
        """RMS of A·sin(ωt) should be A/√2."""
        A = 325.27  # peak voltage for 230 V RMS
        signal = pure_sine(A)
        expected = A / np.sqrt(2)
        assert abs(compute_rms(signal) - expected) < 0.01, (
            f"Expected {expected:.3f}, got {compute_rms(signal):.3f}"
        )

    def test_dc_rms(self):
        """RMS of a constant DC signal should equal its value."""
        dc = np.full(1000, 5.0)
        assert abs(compute_rms(dc) - 5.0) < 1e-9

    def test_zero_rms(self):
        assert compute_rms(np.zeros(100)) == 0.0

    def test_empty_signal(self):
        assert compute_rms(np.array([])) == 0.0

    def test_230v_rms(self):
        """230 V RMS from a 325.27 V peak sine wave."""
        v_peak = 230.0 * np.sqrt(2)
        signal = pure_sine(v_peak)
        assert abs(compute_rms(signal) - 230.0) < 0.5


# ── Frequency estimation tests ─────────────────────────────────────────────

class TestEstimateFrequency:
    def test_50hz(self):
        signal = pure_sine(325.0, 50.0)
        f_est = estimate_frequency(signal, FS, 50.0)
        assert abs(f_est - 50.0) < 0.5, f"Expected ~50 Hz, got {f_est:.3f} Hz"

    def test_60hz(self):
        signal = pure_sine(325.0, 60.0)
        f_est = estimate_frequency(signal, FS, 60.0)
        assert abs(f_est - 60.0) < 0.5, f"Expected ~60 Hz, got {f_est:.3f} Hz"

    def test_off_frequency(self):
        signal = pure_sine(325.0, 51.5)
        f_est = estimate_frequency(signal, FS, 50.0)
        assert abs(f_est - 51.5) < 1.0, f"Expected ~51.5 Hz, got {f_est:.3f} Hz"

    def test_short_signal_fallback(self):
        """Single sample should return nominal frequency."""
        f_est = estimate_frequency(np.array([1.0]), FS, 50.0)
        assert f_est == 50.0


# ── THD tests ──────────────────────────────────────────────────────────────

class TestComputeTHD:
    def test_pure_sine_thd_near_zero(self):
        """A pure sine wave should have THD near 0 %."""
        signal = pure_sine(325.27)
        thd, _ = compute_thd(signal, FS, F1)
        assert thd < 2.0, f"Pure sine THD should be near 0, got {thd:.2f}%"

    def test_harmonic_distortion_detected(self):
        """Adding a 3rd harmonic at 10 % should give THD > 8 %."""
        v1 = 325.27
        signal = pure_sine(v1) + pure_sine(v1 * 0.10, 3 * F1)
        thd, _ = compute_thd(signal, FS, F1)
        assert thd > 8.0, f"Expected THD > 8 %, got {thd:.2f}%"

    def test_thd_harmonics_dict_has_fundamental(self):
        signal = pure_sine(325.27)
        _, harmonics = compute_thd(signal, FS, F1)
        assert 1 in harmonics, "Harmonics dict must include the fundamental (key=1)"

    def test_thd_empty_signal(self):
        thd, _ = compute_thd(np.array([]), FS, F1)
        assert thd == 0.0


# ── Rolling RMS tests ──────────────────────────────────────────────────────

class TestRollingRMS:
    def test_output_length(self):
        sig = np.ones(1000)
        result = rolling_rms(sig, window_samples=50)
        assert len(result) == 1000

    def test_constant_signal(self):
        """Rolling RMS of a constant = constant."""
        sig = np.full(200, 3.0)
        result = rolling_rms(sig, window_samples=10)
        assert np.allclose(result[-100:], 3.0, atol=1e-9)

    def test_window_one(self):
        """Window of 1 should equal abs(signal)."""
        sig = np.array([1.0, -2.0, 3.0])
        result = rolling_rms(sig, window_samples=1)
        expected = np.abs(sig)
        assert np.allclose(result, expected)


# ── Full waveform metrics ──────────────────────────────────────────────────

class TestComputeMetricsFromWaveform:
    def _make_df(self, v_signal, i_signal=None, freq_col=None) -> pd.DataFrame:
        df = pd.DataFrame({
            "timestamp": t,
            "voltage": v_signal,
        })
        if i_signal is not None:
            df["current"] = i_signal
        if freq_col is not None:
            df["frequency"] = freq_col
        return df

    def test_vrms_correct(self):
        v = pure_sine(230.0 * np.sqrt(2))
        df = self._make_df(v)
        m = compute_metrics_from_waveform(df, nominal_voltage=230.0, sampling_frequency=FS)
        assert abs(m["vrms"] - 230.0) < 1.0

    def test_frequency_correct(self):
        v = pure_sine(325.0, 50.0)
        df = self._make_df(v)
        m = compute_metrics_from_waveform(df, nominal_frequency=50.0, sampling_frequency=FS)
        assert abs(m["frequency"] - 50.0) < 0.5

    def test_power_factor_with_current(self):
        v = pure_sine(325.0)
        i = pure_sine(32.5, phase=0.0)   # in-phase → PF ≈ 1
        df = self._make_df(v, i)
        m = compute_metrics_from_waveform(df, sampling_frequency=FS)
        assert m["power_factor"] is not None
        assert abs(m["power_factor"] - 1.0) < 0.05

    def test_no_current_power_is_none(self):
        v = pure_sine(325.0)
        df = self._make_df(v)
        m = compute_metrics_from_waveform(df, sampling_frequency=FS)
        assert m["active_power"] is None
        assert m["power_factor"] is None


# ── Manual metrics ─────────────────────────────────────────────────────────

class TestComputeMetricsFromManual:
    def test_apparent_power(self):
        m = compute_metrics_from_manual(230.0, 10.0, 50.0, 0.95, 3.0)
        assert abs(m["apparent_power"] - 2300.0) < 0.01

    def test_active_power(self):
        m = compute_metrics_from_manual(230.0, 10.0, 50.0, 0.95, 3.0)
        assert abs(m["active_power"] - 2185.0) < 1.0

    def test_unity_pf_zero_reactive(self):
        m = compute_metrics_from_manual(230.0, 10.0, 50.0, 1.0, 0.0)
        assert abs(m["reactive_power"]) < 1.0

    def test_source_label(self):
        m = compute_metrics_from_manual(230.0, 10.0, 50.0, 0.95, 3.0)
        assert m["source"] == "manual"
