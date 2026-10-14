"""
app.py — Home Dashboard
AI-Based Power Quality Monitoring System
"""

import streamlit as st
import pandas as pd
import numpy as np

# ---------- Page configuration ----------
st.set_page_config(
    page_title=" A.E.G.I.S. The AI Power Quality Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563EB 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: #f7f9fc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .status-normal   { color: #16a34a; font-weight: bold; }
    .status-warning  { color: #d97706; font-weight: bold; }
    .status-critical { color: #dc2626; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2rem;">⚡ AI-Based Power Quality Monitoring System</h1>
    <p style="margin:0.5rem 0 0 0; font-size:1.05rem; opacity:0.9;">
        Intelligent Analysis, Disturbance Detection and Electrical Parameter Monitoring
    </p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar navigation ----------
st.sidebar.title("⚡ Navigation")
st.sidebar.markdown("Use the pages below to explore the application.")
st.sidebar.info(
    "**Workflow:** Data Input → Electrical Analysis → Disturbance Detection "
    "→ Machine Learning → Reports"
)

# ---------- Session state defaults ----------
if "data" not in st.session_state:
    st.session_state["data"] = None
if "metrics" not in st.session_state:
    st.session_state["metrics"] = None
if "disturbances" not in st.session_state:
    st.session_state["disturbances"] = None
if "ml_model_info" not in st.session_state:
    st.session_state["ml_model_info"] = None
if "ml_prediction" not in st.session_state:
    st.session_state["ml_prediction"] = None
if "nominal_voltage" not in st.session_state:
    st.session_state["nominal_voltage"] = 230.0
if "nominal_frequency" not in st.session_state:
    st.session_state["nominal_frequency"] = 50.0
if "dataset_description" not in st.session_state:
    st.session_state["dataset_description"] = "No data loaded"

# ---------- Key metrics display ----------
metrics = st.session_state.get("metrics")

st.subheader("📊 Live Metric Overview")

col1, col2, col3, col4, col5, col6 = st.columns(6)

def _val(m, key, fmt=""):
    if m is None:
        return "—"
    v = m.get(key)
    if v is None:
        return "N/A"
    if fmt:
        return f"{v:{fmt}}"
    return str(v)

with col1:
    st.metric("RMS Voltage (V)",   _val(metrics, "vrms", ".2f"))
with col2:
    st.metric("RMS Current (A)",   _val(metrics, "irms", ".3f"))
with col3:
    st.metric("Frequency (Hz)",    _val(metrics, "frequency", ".3f"))
with col4:
    st.metric("Power Factor",      _val(metrics, "power_factor", ".3f"))
with col5:
    thd = metrics.get("thd_pct") if metrics else None
    thd_str = f"{thd:.2f}%" if thd is not None else "N/A"
    st.metric("THD (%)", thd_str)
with col6:
    disturbances = st.session_state.get("disturbances")
    if disturbances is None:
        status_str = "No data"
        status_class = ""
    elif isinstance(disturbances, pd.DataFrame):
        status_str = "Normal ✅" if len(disturbances) == 0 else f"{len(disturbances)} event(s) ⚠️"
        status_class = "status-normal" if len(disturbances) == 0 else "status-warning"
    else:
        status_str = str(disturbances)
        status_class = ""
    st.metric("Power Quality", status_str)

# ---------- Quick start guide ----------
st.markdown("---")
st.subheader("🚀 Quick Start Guide")

cols = st.columns(4)
steps = [
    ("1️⃣  Load Data",    "pages/1_Data_Input.py",        "Go to **Data Input** to generate, upload, or enter electrical data."),
    ("2️⃣  Analyse",      "pages/2_Electrical_Analysis.py","Go to **Electrical Analysis** for RMS, THD, and power calculations."),
    ("3️⃣  Detect",       "pages/3_Disturbance_Detection.py","Go to **Disturbance Detection** for rule-based event identification."),
    ("4️⃣  ML Classify",  "pages/4_Machine_Learning.py",   "Go to **Machine Learning** for AI-based condition classification."),
]

for col, (title, _, desc) in zip(cols, steps):
    with col:
        st.markdown(f"**{title}**")
        st.caption(desc)

# ---------- System overview ----------
st.markdown("---")
st.subheader("🏗️ System Architecture Overview")
st.markdown("""
```
Electrical Data
      │
      ▼
  Data Validation
      │
      ▼
  Signal Processing  ──→  RMS · Frequency · FFT · THD
      │
      ▼
  Electrical Parameter Calculation
      │
      ├──→  Rule-Based Disturbance Detection
      │
      ├──→  Machine Learning Classification
      │
      ├──→  Interactive Visualizations
      │
      └──→  Report Generation & Downloads
```
""")

st.caption("⚠️ This application is designed for educational and demonstration purposes. "
           "All results should be verified by a qualified electrical engineer before "
           "any equipment or protection settings are changed.")
