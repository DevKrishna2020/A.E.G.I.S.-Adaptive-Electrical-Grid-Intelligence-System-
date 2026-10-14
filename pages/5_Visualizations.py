"""
5_Visualizations.py — Interactive Visualizations page
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
from modules.visualization import (
    plot_voltage_waveform,
    plot_current_waveform,
    plot_rms_trend,
    plot_frequency_trend,
    plot_harmonic_spectrum,
    plot_disturbance_timeline,
    plot_waveform_comparison,
    plot_confusion_matrix,
)
from modules.signal_processing import rolling_rms
from modules.data_loader import generate_synthetic_data

st.set_page_config(page_title="Visualizations", page_icon="📈", layout="wide")
st.title("📈 Interactive Visualizations")
st.markdown("All charts are interactive — zoom, pan, and hover to inspect data points.")

data: pd.DataFrame | None = st.session_state.get("data")
metrics   = st.session_state.get("metrics")
events_df = st.session_state.get("disturbances")
ml_info   = st.session_state.get("ml_model_info")
nom_v     = float(st.session_state.get("nominal_voltage",  230.0))
nom_f     = float(st.session_state.get("nominal_frequency", 50.0))
fs        = float(st.session_state.get("sampling_frequency", 3200.0))

if data is None:
    st.warning("⚠️ No data loaded. Go to **Data Input** first.")
    st.stop()

# ── 1. Voltage waveform ───────────────────────────────────
st.subheader("1. Voltage Waveform")
st.plotly_chart(plot_voltage_waveform(data), use_container_width=True)

# ── 2. Current waveform ───────────────────────────────────
st.subheader("2. Current Waveform")
st.plotly_chart(plot_current_waveform(data), use_container_width=True)

# ── 3. RMS Voltage trend ──────────────────────────────────
st.subheader("3. RMS Voltage Trend")
voltage = data["voltage"].to_numpy(dtype=float)
timestamps = data["timestamp"].to_numpy(dtype=float) if "timestamp" in data.columns else (
    np.arange(len(voltage)) / fs
)
w = max(1, int(fs / nom_f))   # 1-cycle window
vrms_roll = rolling_rms(voltage, w)
st.plotly_chart(plot_rms_trend(timestamps, vrms_roll, nom_v), use_container_width=True)

# ── 4. Frequency variation ────────────────────────────────
st.subheader("4. Frequency Variation")
st.plotly_chart(plot_frequency_trend(data, nom_f), use_container_width=True)

# ── 5. Harmonic spectrum / THD ────────────────────────────
st.subheader("5. Harmonic Spectrum (Voltage THD)")
harmonics = (metrics or {}).get("thd_harmonics", {})
freq_f1   = (metrics or {}).get("frequency", nom_f)
if harmonics:
    st.plotly_chart(plot_harmonic_spectrum(harmonics, freq_f1), use_container_width=True)
else:
    st.info("ℹ️ Run **Electrical Analysis** to compute the harmonic spectrum.")

# ── 6. Disturbance timeline ───────────────────────────────
st.subheader("6. Power Quality Disturbance Timeline")
if events_df is not None:
    st.plotly_chart(plot_disturbance_timeline(events_df), use_container_width=True)
else:
    st.info("ℹ️ Run **Disturbance Detection** first to display the timeline.")

# ── 7. Waveform comparison ────────────────────────────────
st.subheader("7. Waveform Comparison: Normal vs Disturbed")
st.markdown(
    "Select a disturbance scenario to generate a comparison reference waveform "
    "(shown against the currently loaded data)."
)
comp_scenario = st.selectbox("Comparison scenario", [
    "voltage_sag", "voltage_swell", "interruption",
    "harmonic_distortion", "frequency_deviation",
], key="comp_scenario")
comp_severity = st.slider("Comparison severity", 0.1, 1.0, 0.5, 0.1, key="comp_sev")

df_normal = generate_synthetic_data(
    scenario="normal", nominal_voltage=nom_v,
    nominal_frequency=nom_f, sampling_frequency=fs,
    duration=min(float(len(data)) / fs, 0.1),
)
# Trim both to the same length
n_pts = min(len(data), len(df_normal))
df_n = df_normal.iloc[:n_pts].reset_index(drop=True)
df_d = data.iloc[:n_pts].reset_index(drop=True)

st.plotly_chart(
    plot_waveform_comparison(df_n, df_d, label=comp_scenario.replace("_", " ").title()),
    use_container_width=True,
)

# ── 8. ML Confusion matrix ───────────────────────────────
st.subheader("8. ML Classifier — Confusion Matrix")
if ml_info:
    st.plotly_chart(
        plot_confusion_matrix(ml_info["confusion_matrix"], ml_info["class_names"]),
        use_container_width=True,
    )
else:
    st.info("ℹ️ Train the model in **Machine Learning** to display the confusion matrix.")
