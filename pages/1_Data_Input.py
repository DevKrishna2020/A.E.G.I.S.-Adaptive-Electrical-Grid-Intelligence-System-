"""
1_Data_Input.py — Data Input and Simulation page
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from modules.data_loader import generate_synthetic_data, load_csv, get_sample_csv

st.set_page_config(page_title="Data Input", page_icon="📂", layout="wide")
st.title("📂 Data Input and Simulation")
st.markdown("Choose one of the three methods below to load electrical measurement data.")

# Sidebar: nominal parameters (shared across pages)
st.sidebar.header("⚙️ System Parameters")
nom_v = st.sidebar.number_input("Nominal RMS Voltage (V)", min_value=10.0, max_value=1000.0,
                                 value=float(st.session_state.get("nominal_voltage", 230.0)), step=1.0)
nom_f = st.sidebar.number_input("Nominal Frequency (Hz)", min_value=40.0, max_value=70.0,
                                 value=float(st.session_state.get("nominal_frequency", 50.0)), step=1.0)
st.session_state["nominal_voltage"]  = nom_v
st.session_state["nominal_frequency"] = nom_f

# ──────────────────────────────────────────────────────────
# Tab layout
# ──────────────────────────────────────────────────────────
tab_sim, tab_csv, tab_manual = st.tabs(["🔬 Option A: Simulation", "📁 Option B: CSV Upload", "✏️ Option C: Manual Input"])


# ======================================================
# OPTION A: Simulation
# ======================================================
with tab_sim:
    st.subheader("Generate Synthetic Electrical Waveform")
    st.markdown("Configure the simulation parameters and click **Generate Electrical Data**.")

    c1, c2 = st.columns(2)
    with c1:
        scenario = st.selectbox("Scenario / Disturbance Type", [
            "normal", "voltage_sag", "voltage_swell", "interruption",
            "harmonic_distortion", "frequency_deviation", "multiple",
        ], format_func=lambda x: x.replace("_", " ").title())
        fs = st.number_input("Sampling Frequency (Hz)", min_value=200.0, max_value=50000.0,
                             value=3200.0, step=200.0)
        duration = st.number_input("Duration (s)", min_value=0.02, max_value=1.0,
                                   value=0.1, step=0.01)
    with c2:
        severity = st.slider("Disturbance Severity", 0.0, 1.0, 0.5, 0.05,
                             help="0 = minimal, 1 = maximum disturbance")
        seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1)

    st.info(
        f"**Parameters:** Nominal Voltage = {nom_v} V | Nominal Frequency = {nom_f} Hz | "
        f"Sampling Frequency = {fs} Hz | Duration = {duration} s | Severity = {severity}"
    )

    if st.button("⚡ Generate Electrical Data", type="primary"):
        with st.spinner("Generating waveform..."):
            df = generate_synthetic_data(
                scenario=scenario,
                nominal_voltage=nom_v,
                nominal_frequency=nom_f,
                sampling_frequency=fs,
                duration=duration,
                severity=severity,
                random_seed=int(seed),
            )
        st.session_state["data"] = df
        st.session_state["sampling_frequency"] = fs
        st.session_state["dataset_description"] = (
            f"Simulated — {scenario.replace('_', ' ').title()} | "
            f"Vnom={nom_v}V | f={nom_f}Hz | fs={fs}Hz | duration={duration}s"
        )
        # Clear downstream results so they're recalculated
        for key in ("metrics", "disturbances", "ml_prediction"):
            st.session_state[key] = None
        st.success(f"✅ Generated {len(df):,} samples. Proceed to **Electrical Analysis**.")
        st.dataframe(df.head(20), use_container_width=True)


# ======================================================
# OPTION B: CSV Upload
# ======================================================
with tab_csv:
    st.subheader("Upload an Electrical Measurement CSV")
    st.markdown("""
**Required column:** `voltage`

**Optional columns:** `timestamp`, `current`, `frequency`

Each row should contain one instantaneous sample.
    """)

    # Sample download
    sample_csv = get_sample_csv()
    st.download_button(
        "⬇️  Download Sample CSV",
        data=sample_csv,
        file_name="sample_power_quality.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        df, msg = load_csv(uploaded)
        if df is None:
            st.error(f"❌ {msg}")
        else:
            if msg:
                st.warning(msg)
            st.session_state["data"] = df
            fs_guess = 3200.0
            if "timestamp" in df.columns and len(df) > 1:
                dt = float(df["timestamp"].iloc[1] - df["timestamp"].iloc[0])
                fs_guess = 1.0 / dt if dt > 0 else 3200.0
            st.session_state["sampling_frequency"] = fs_guess
            st.session_state["dataset_description"] = f"CSV Upload: {uploaded.name}"
            for key in ("metrics", "disturbances", "ml_prediction"):
                st.session_state[key] = None
            st.success(f"✅ Loaded {len(df):,} rows from '{uploaded.name}'. Proceed to **Electrical Analysis**.")
            st.dataframe(df.head(20), use_container_width=True)


# ======================================================
# OPTION C: Manual Input
# ======================================================
with tab_manual:
    st.subheader("Enter Summary Electrical Measurements")
    st.info(
        "Manual input performs a rule-based power quality assessment from "
        "summary statistics only. Waveform-level analysis (THD from FFT, "
        "frequency estimation) is NOT performed — the values you enter are "
        "used directly."
    )

    mc1, mc2 = st.columns(2)
    with mc1:
        m_vrms = st.number_input("RMS Voltage (V)", min_value=0.0, max_value=10000.0,
                                  value=nom_v, step=1.0)
        m_irms = st.number_input("RMS Current (A)", min_value=0.0, max_value=10000.0,
                                  value=23.0, step=0.1)
        m_freq = st.number_input("Frequency (Hz)", min_value=0.0, max_value=200.0,
                                  value=nom_f, step=0.1)
    with mc2:
        m_pf  = st.number_input("Power Factor (0–1)", min_value=0.0, max_value=1.0,
                                 value=0.95, step=0.01)
        m_thd = st.number_input("Voltage THD (%)", min_value=0.0, max_value=100.0,
                                 value=3.0, step=0.1)

    if st.button("📊 Analyse Manual Input", type="primary"):
        from modules.electrical_metrics import compute_metrics_from_manual
        from modules.disturbance_detection import classify_snapshot

        metrics = compute_metrics_from_manual(m_vrms, m_irms, m_freq, m_pf, m_thd)
        result  = classify_snapshot(
            vrms=m_vrms,
            nominal_voltage=nom_v,
            frequency=m_freq,
            nominal_frequency=nom_f,
            thd_pct=m_thd,
        )

        st.session_state["metrics"] = metrics
        st.session_state["disturbances"] = None  # no time-series for manual
        st.session_state["snapshot_result"] = result
        st.session_state["dataset_description"] = (
            f"Manual Input: Vrms={m_vrms}V, Irms={m_irms}A, f={m_freq}Hz, PF={m_pf}, THD={m_thd}%"
        )

        # Display results inline
        st.success("✅ Manual analysis complete.")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("RMS Voltage",  f"{m_vrms:.2f} V")
        rc2.metric("RMS Current",  f"{m_irms:.3f} A")
        rc3.metric("Frequency",    f"{m_freq:.3f} Hz")

        rca, rcb, rcc = st.columns(3)
        rca.metric("Power Factor", f"{m_pf:.3f}")
        rcb.metric("THD",          f"{m_thd:.2f}%")
        rcc.metric("Active Power", f"{metrics['active_power']:.2f} W")

        st.markdown("**Power Quality Status:**")
        if result["disturbances"]:
            for d in result["disturbances"]:
                st.warning(f"⚠️ {d}")
        else:
            st.success("✅ Normal — No disturbances detected.")
