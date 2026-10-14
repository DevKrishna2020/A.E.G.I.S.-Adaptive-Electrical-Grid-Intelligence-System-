"""
3_Disturbance_Detection.py — Rule-Based Disturbance Detection page
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from modules.disturbance_detection import (
    classify_snapshot,
    detect_disturbances_timeseries,
    DEFAULT_THRESHOLDS,
)
from modules.visualization import plot_rms_trend, plot_disturbance_timeline
from modules.signal_processing import rolling_rms

st.set_page_config(page_title="Disturbance Detection", page_icon="🔍", layout="wide")
st.title("🔍 Power Quality Disturbance Detection")
st.markdown("Rule-based detection using configurable engineering thresholds.")

data: pd.DataFrame | None = st.session_state.get("data")
metrics = st.session_state.get("metrics")
nom_v   = float(st.session_state.get("nominal_voltage",  230.0))
nom_f   = float(st.session_state.get("nominal_frequency", 50.0))
fs      = float(st.session_state.get("sampling_frequency", 3200.0))

if data is None:
    st.warning("⚠️ No waveform data loaded. Go to **Data Input** first.")
    st.stop()

# ── Configurable thresholds ──────────────────────────────
with st.expander("⚙️  Configure Detection Thresholds", expanded=False):
    st.markdown(
        "Defaults follow IEC 61000 / EN 50160 conventions for LV systems. "
        "Adjust if your system uses a different standard."
    )
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        thr_sag   = st.number_input("Sag threshold (p.u.)",          0.01, 0.99, 0.90, 0.01)
        thr_swell = st.number_input("Swell threshold (p.u.)",         1.01, 2.00, 1.10, 0.01)
        thr_int   = st.number_input("Interruption threshold (p.u.)",  0.01, 0.50, 0.10, 0.01)
    with tc2:
        thr_freq  = st.number_input("Frequency deviation limit (Hz)", 0.1,  5.0,  0.5,  0.1)
        thr_thd_w = st.number_input("THD warning threshold (%)",      1.0,  20.0, 5.0,  0.5)
        thr_thd_h = st.number_input("THD high threshold (%)",         2.0,  50.0, 8.0,  0.5)
    with tc3:
        win_cycles = st.number_input("Rolling RMS window (cycles)",   1,    20,   1,    1)

thresholds = {
    "sag_threshold":         thr_sag,
    "swell_threshold":       thr_swell,
    "interruption_threshold":thr_int,
    "freq_deviation_hz":     thr_freq,
    "thd_warning_pct":       thr_thd_w,
    "thd_high_pct":          thr_thd_h,
}

# ── Snapshot classification ───────────────────────────────
st.markdown("---")
st.subheader("Snapshot Classification")
st.caption("Classification of the overall dataset using mean/aggregate metrics.")

if metrics:
    vrms  = metrics.get("vrms", nom_v)
    freq  = metrics.get("frequency", nom_f)
    thd   = metrics.get("thd_pct")

    snap  = classify_snapshot(vrms, nom_v, freq, nom_f, thd, thresholds)
    details = snap["details"]

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Voltage Ratio (p.u.)", f"{details['voltage_ratio']:.4f}")
    sc2.metric("Freq Deviation (Hz)", f"{details['frequency_deviation_hz']:.4f}")
    sc3.metric("Voltage Status", details["voltage_status"])

    sc4, sc5 = st.columns(2)
    sc4.metric("Frequency Status", details["frequency_status"])
    sc5.metric("THD Status", details.get("thd_status", "N/A"))

    if snap["disturbances"]:
        for d in snap["disturbances"]:
            st.warning(f"⚠️ **{d}** detected.")
    else:
        st.success("✅ **Normal operating conditions** — No disturbances detected in snapshot.")
else:
    st.info("Run **Electrical Analysis** first to see snapshot results.")

# ── Time-series detection ─────────────────────────────────
st.markdown("---")
st.subheader("Time-Series Disturbance Detection")

if st.button("🔍 Run Time-Series Detection", type="primary"):
    with st.spinner("Analysing waveform over time..."):
        events_df = detect_disturbances_timeseries(
            data,
            nominal_voltage=nom_v,
            nominal_frequency=nom_f,
            sampling_frequency=fs,
            thresholds=thresholds,
            window_cycles=int(win_cycles),
        )
    st.session_state["disturbances"] = events_df

events_df = st.session_state.get("disturbances")

if events_df is not None:
    if len(events_df) == 0:
        st.success("✅ No disturbance events detected in the time series.")
    else:
        st.warning(f"⚠️ **{len(events_df)}** disturbance event(s) detected.")
        st.dataframe(events_df, use_container_width=True)

    # Rolling RMS chart
    st.subheader("Rolling RMS Voltage")
    voltage = data["voltage"].to_numpy(dtype=float)
    timestamps = data["timestamp"].to_numpy(dtype=float) if "timestamp" in data.columns else (
        np.arange(len(voltage)) / fs
    )
    w = max(1, int(int(win_cycles) * fs / nom_f))
    vrms_roll = rolling_rms(voltage, w)
    st.plotly_chart(
        plot_rms_trend(timestamps, vrms_roll, nom_v, thresholds),
        use_container_width=True,
    )

    # Timeline
    st.subheader("Disturbance Timeline")
    st.plotly_chart(plot_disturbance_timeline(events_df), use_container_width=True)
else:
    st.info("Click **Run Time-Series Detection** to analyse the waveform for disturbance events.")

# ── Threshold documentation ───────────────────────────────
st.markdown("---")
with st.expander("📖  Threshold Reference and Standard Notes"):
    st.markdown("""
| Parameter | Default | Standard Reference |
|---|---|---|
| Voltage sag | < 0.90 p.u. | EN 50160 / IEC 61000-4-30 |
| Voltage swell | > 1.10 p.u. | EN 50160 / IEC 61000-4-30 |
| Voltage interruption | < 0.10 p.u. | EN 50160 |
| Frequency deviation | ±0.5 Hz | EN 50160 interconnected system |
| THD warning | 5 % | Conservative (EN 50160 LV limit: 8 %) |
| THD high | 8 % | EN 50160 LV limit |

**Note:** Different electrical standards (IEEE 1159, IEC 61000, EN 50160, ANSI C84.1) use
different thresholds and event duration criteria. These defaults are indicative only.
Always verify the applicable standard for your region and application.
    """)
