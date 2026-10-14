"""
test_disturbance_detection.py
------------------------------
Automated tests for rule-based disturbance detection.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from modules.disturbance_detection import (
    classify_snapshot,
    detect_disturbances_timeseries,
    DEFAULT_THRESHOLDS,
)
from modules.data_loader import generate_synthetic_data


FS   = 3200.0
NOM_V = 230.0
NOM_F = 50.0


# ── classify_snapshot tests ────────────────────────────────────────────────

class TestClassifySnapshot:
    def test_normal_conditions(self):
        result = classify_snapshot(230.0, NOM_V, 50.0, NOM_F, 2.0)
        assert result["status"] == "Normal"
        assert result["disturbances"] == []

    def test_voltage_sag_detected(self):
        result = classify_snapshot(180.0, NOM_V, 50.0, NOM_F, 2.0)
        assert "Voltage Sag" in result["disturbances"]

    def test_voltage_swell_detected(self):
        result = classify_snapshot(265.0, NOM_V, 50.0, NOM_F, 2.0)
        assert "Voltage Swell" in result["disturbances"]

    def test_interruption_detected(self):
        result = classify_snapshot(10.0, NOM_V, 50.0, NOM_F, 2.0)
        assert "Voltage Interruption" in result["disturbances"]

    def test_interruption_not_sag(self):
        """An interruption should NOT also be classified as a sag."""
        result = classify_snapshot(10.0, NOM_V, 50.0, NOM_F, 2.0)
        assert "Voltage Sag" not in result["disturbances"]

    def test_frequency_deviation_detected(self):
        result = classify_snapshot(230.0, NOM_V, 52.0, NOM_F, 2.0)
        assert "Frequency Deviation" in result["disturbances"]

    def test_frequency_normal_within_band(self):
        result = classify_snapshot(230.0, NOM_V, 50.3, NOM_F, 2.0)
        assert "Frequency Deviation" not in result["disturbances"]

    def test_high_thd_detected(self):
        result = classify_snapshot(230.0, NOM_V, 50.0, NOM_F, 12.0)
        assert any("Harmonic Distortion" in d for d in result["disturbances"])

    def test_no_thd_no_thd_disturbance(self):
        result = classify_snapshot(230.0, NOM_V, 50.0, NOM_F, None)
        assert not any("Harmonic" in d for d in result["disturbances"])

    def test_custom_thresholds(self):
        """Tighter sag threshold (0.95) should catch a 0.92 p.u. voltage."""
        custom = {"sag_threshold": 0.95}
        result = classify_snapshot(210.0, NOM_V, 50.0, NOM_F, 2.0, thresholds=custom)
        assert "Voltage Sag" in result["disturbances"]

    def test_voltage_ratio_in_details(self):
        result = classify_snapshot(230.0, NOM_V, 50.0, NOM_F, 2.0)
        assert "voltage_ratio" in result["details"]
        assert abs(result["details"]["voltage_ratio"] - 1.0) < 1e-6


# ── detect_disturbances_timeseries tests ───────────────────────────────────

class TestDetectDisturbancesTimeseries:
    def _make_sag_df(self, severity: float = 0.5) -> pd.DataFrame:
        return generate_synthetic_data(
            scenario="voltage_sag",
            nominal_voltage=NOM_V,
            nominal_frequency=NOM_F,
            sampling_frequency=FS,
            duration=0.1,
            severity=severity,
            random_seed=1,
        )

    def _make_normal_df(self) -> pd.DataFrame:
        return generate_synthetic_data(
            scenario="normal",
            nominal_voltage=NOM_V,
            nominal_frequency=NOM_F,
            sampling_frequency=FS,
            duration=0.1,
            random_seed=1,
        )

    def test_normal_waveform_no_events(self):
        df = self._make_normal_df()
        events = detect_disturbances_timeseries(df, NOM_V, NOM_F, FS)
        assert len(events) == 0, f"Expected 0 events, got {len(events)}"

    def test_sag_waveform_produces_events(self):
        df = self._make_sag_df(0.6)
        events = detect_disturbances_timeseries(df, NOM_V, NOM_F, FS)
        assert len(events) > 0, "Expected disturbance events for a sag waveform"

    def test_sag_event_type(self):
        df = self._make_sag_df(0.6)
        events = detect_disturbances_timeseries(df, NOM_V, NOM_F, FS)
        types = events["disturbance_type"].str.lower().tolist()
        assert any("sag" in t for t in types), f"Expected sag event, got: {types}"

    def test_events_df_columns(self):
        df = self._make_sag_df(0.6)
        events = detect_disturbances_timeseries(df, NOM_V, NOM_F, FS)
        required_cols = {"disturbance_type", "start_time", "end_time", "duration_s"}
        assert required_cols.issubset(set(events.columns)), (
            f"Missing columns: {required_cols - set(events.columns)}"
        )

    def test_interruption_detected(self):
        df = generate_synthetic_data(
            scenario="interruption",
            nominal_voltage=NOM_V,
            nominal_frequency=NOM_F,
            sampling_frequency=FS,
            duration=0.1,
            severity=1.0,
            random_seed=2,
        )
        events = detect_disturbances_timeseries(df, NOM_V, NOM_F, FS)
        assert len(events) > 0
        types = events["disturbance_type"].str.lower().tolist()
        assert any("interrupt" in t for t in types), f"Expected interruption, got: {types}"

    def test_swell_detected(self):
        df = generate_synthetic_data(
            scenario="voltage_swell",
            nominal_voltage=NOM_V,
            nominal_frequency=NOM_F,
            sampling_frequency=FS,
            duration=0.1,
            severity=0.8,
            random_seed=3,
        )
        events = detect_disturbances_timeseries(df, NOM_V, NOM_F, FS)
        assert len(events) > 0
        types = events["disturbance_type"].str.lower().tolist()
        assert any("swell" in t for t in types), f"Expected swell, got: {types}"

    def test_duration_positive(self):
        df = self._make_sag_df(0.6)
        events = detect_disturbances_timeseries(df, NOM_V, NOM_F, FS)
        if len(events) > 0:
            assert (events["duration_s"] >= 0).all()

    def test_severity_in_0_1_range(self):
        df = self._make_sag_df(0.6)
        events = detect_disturbances_timeseries(df, NOM_V, NOM_F, FS)
        if len(events) > 0:
            assert (events["severity"] >= 0).all()
            assert (events["severity"] <= 1).all()


# ── Default thresholds sanity check ──────────────────────────────────────

class TestDefaultThresholds:
    def test_sag_lt_swell(self):
        assert DEFAULT_THRESHOLDS["sag_threshold"] < DEFAULT_THRESHOLDS["swell_threshold"]

    def test_interruption_lt_sag(self):
        assert DEFAULT_THRESHOLDS["interruption_threshold"] < DEFAULT_THRESHOLDS["sag_threshold"]

    def test_thd_warning_lt_high(self):
        assert DEFAULT_THRESHOLDS["thd_warning_pct"] < DEFAULT_THRESHOLDS["thd_high_pct"]
