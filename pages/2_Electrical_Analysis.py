"""
2_Electrical_Analysis.py — Electrical Parameter Analysis page
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Electrical Analysis", page_icon="🔌", layout="wide")
st.title("🔌 Electrical Parameter Analysis")

data: pd.DataFrame | None = st.session_state.get("data")
nom_v  = float(st.session_state.get("nominal_voltage",  230.0))
nom_f  = float(st.session_state.get("nominal_frequency", 50.0))
fs     = float(st.session_state.get("sampling_frequency", 3200.0))

if data is None:
    st.warning("⚠️ No waveform data loaded. Go to **Data Input** first.")
    st.stop()

# ── Calculate metrics ─────────────────────────────────────
from modules.electrical_metrics import compute_metrics_from_waveform, metrics_summary

if st.session_state.get("metrics") is None or st.button("🔄 Recalculate"):
    with st.spinner("Calculating electrical parameters..."):
        m = compute_metrics_from_waveform(
            data, nominal_voltage=nom_v,
            nominal_frequency=nom_f,
            sampling_frequency=fs,
        )
    st.session_state["metrics"] = m

m = st.session_state["metrics"]

# ── Metric cards ─────────────────────────────────────────
st.subheader("Calculated Electrical Metrics")

c1, c2, c3 = st.columns(3)
c4, c5, c6 = st.columns(3)

with c1:
    st.metric("RMS Voltage (V)", f"{m['vrms']:.3f}" if m.get("vrms") is not None else "N/A")
with c2:
    st.metric("RMS Current (A)", f"{m['irms']:.4f}" if m.get("irms") is not None else "N/A")
with c3:
    st.metric("Frequency (Hz)", f"{m['frequency']:.3f}" if m.get("frequency") is not None else "N/A")
with c4:
    st.metric("Power Factor", f"{m['power_factor']:.4f}" if m.get("power_factor") is not None else "N/A")
with c5:
    thd = m.get("thd_pct")
    st.metric("Voltage THD (%)", f"{thd:.3f}" if thd is not None else "Insufficient data")
with c6:
    ap = m.get("active_power")
    st.metric("Active Power (W)", f"{ap:.2f}" if ap is not None else "N/A")

st.markdown("---")
c7, c8 = st.columns(2)
with c7:
    sp = m.get("apparent_power")
    st.metric("Apparent Power (VA)", f"{sp:.2f}" if sp is not None else "N/A")
with c8:
    rp = m.get("reactive_power")
    st.metric("Reactive Power (VAR)", f"{rp:.2f}" if rp is not None else "N/A")

# ── Warnings / Notes ────────────────────────────────────
st.markdown("---")
st.subheader("Notes and Limitations")

if m.get("nyquist_warning"):
    st.warning(f"📡 **Nyquist / Sampling Warning:** {m['nyquist_warning']}")

if not m.get("thd_valid"):
    st.info(f"ℹ️ **THD:** {m.get('thd_note', 'Insufficient waveform data for THD calculation.')}")
else:
    if m.get("thd_note"):
        st.info(f"ℹ️ **THD Note:** {m['thd_note']}")

if m.get("power_note"):
    st.info(f"⚙️ **Power Note:** {m['power_note']}")

if not m.get("has_current"):
    st.warning("⚠️ No current data found. Current-based metrics (active/reactive/apparent power, power factor) are not available.")

# ── THD Harmonic Table ───────────────────────────────────
harmonics = m.get("thd_harmonics", {})
if harmonics:
    st.markdown("---")
    st.subheader("Harmonic Components")
    freq = m.get("frequency", nom_f)
    rows = []
    v1 = harmonics.get(1, 0.0)
    for h, amp in sorted(harmonics.items()):
        pct = 100.0 * amp / v1 if v1 > 1e-9 else 0.0
        rows.append({"Order": h, "Frequency (Hz)": f"{h*freq:.1f}",
                     "RMS Amplitude (V)": f"{amp:.4f}",
                     "Relative to V1 (%)": f"{pct:.2f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── Waveform preview ─────────────────────────────────────
st.markdown("---")
st.subheader("Waveform Preview")
from modules.visualization import plot_voltage_waveform, plot_current_waveform
st.plotly_chart(plot_voltage_waveform(data), use_container_width=True)
if m.get("has_current"):
    st.plotly_chart(plot_current_waveform(data), use_container_width=True)
