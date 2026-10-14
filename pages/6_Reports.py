"""
6_Reports.py — Reports and Downloads page
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
from modules.report_generator import generate_pdf_report, generate_csv_report
from modules.recommendations import get_recommendations, DISCLAIMER

st.set_page_config(page_title="Reports & Downloads", page_icon="📄", layout="wide")
st.title("📄 Reports and Downloads")

metrics       = st.session_state.get("metrics")
events_df     = st.session_state.get("disturbances")
ml_prediction = st.session_state.get("ml_prediction")
ml_info       = st.session_state.get("ml_model_info")
nom_v         = float(st.session_state.get("nominal_voltage",  230.0))
nom_f         = float(st.session_state.get("nominal_frequency", 50.0))
dataset_desc  = st.session_state.get("dataset_description", "Unknown dataset")

if metrics is None:
    st.warning("⚠️ No analysis results found. Complete the **Electrical Analysis** step first.")
    st.stop()

# ── Detected disturbances ─────────────────────────────────
disturbance_list: list[str] = []

if events_df is not None and len(events_df) > 0:
    disturbance_list = list(events_df["disturbance_type"].unique())
elif st.session_state.get("snapshot_result"):
    disturbance_list = st.session_state["snapshot_result"].get("disturbances", [])

# ── Recommendations ───────────────────────────────────────
recommendations = get_recommendations(disturbance_list)

st.subheader("📋 Recommendations")
st.warning(DISCLAIMER)

for rec in recommendations:
    severity = rec.get("severity_level", "info")
    icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(severity, "ℹ️")
    with st.expander(f"{icon} {rec.get('disturbance', 'General')} — {rec.get('summary', '')}"):
        for action in rec.get("actions", []):
            st.markdown(f"- {action}")

# ── Report summary ────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Analysis Summary")

sum_c1, sum_c2 = st.columns(2)
with sum_c1:
    st.markdown("**Electrical Metrics**")
    summary_rows = [
        ("RMS Voltage (V)",   metrics.get("vrms")),
        ("RMS Current (A)",   metrics.get("irms")),
        ("Frequency (Hz)",    metrics.get("frequency")),
        ("Power Factor",      metrics.get("power_factor")),
        ("Active Power (W)",  metrics.get("active_power")),
        ("Voltage THD (%)",   metrics.get("thd_pct")),
    ]
    for label, val in summary_rows:
        v_str = f"{val:.4g}" if val is not None else "N/A"
        st.markdown(f"- **{label}:** {v_str}")

with sum_c2:
    st.markdown("**Detected Disturbances**")
    if disturbance_list:
        for d in disturbance_list:
            st.markdown(f"- ⚠️ {d}")
    else:
        st.markdown("- ✅ Normal — No disturbances detected")

    if ml_prediction:
        st.markdown("**ML Classification**")
        st.markdown(f"- Predicted: **{ml_prediction.get('predicted_class')}**")
        st.markdown(f"- Confidence: **{ml_prediction.get('confidence', 0)*100:.1f}%**")

# ── Download buttons ──────────────────────────────────────
st.markdown("---")
st.subheader("⬇️  Download Reports")

# Build ML results dict for PDF
ml_results_for_report = None
if ml_prediction and ml_info:
    ml_results_for_report = {**ml_prediction, **{k: ml_info[k] for k in ("accuracy", "precision", "recall", "f1")}}

dl_c1, dl_c2 = st.columns(2)

with dl_c1:
    st.markdown("**CSV Report**")
    csv_bytes = generate_csv_report(metrics, events_df, recommendations)
    st.download_button(
        label="⬇️  Download CSV Report",
        data=csv_bytes,
        file_name="power_quality_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

with dl_c2:
    st.markdown("**PDF Report**")
    if st.button("🔄 Generate PDF Report", use_container_width=True):
        with st.spinner("Generating PDF..."):
            try:
                pdf_bytes = generate_pdf_report(
                    metrics=metrics,
                    disturbances_df=events_df,
                    recommendations=recommendations,
                    ml_results=ml_results_for_report,
                    dataset_description=dataset_desc,
                    nominal_voltage=nom_v,
                    nominal_frequency=nom_f,
                )
                st.session_state["pdf_bytes"] = pdf_bytes
                st.success("✅ PDF generated successfully.")
            except Exception as e:
                st.error(f"❌ PDF generation failed: {e}")

    if st.session_state.get("pdf_bytes"):
        st.download_button(
            label="⬇️  Download PDF Report",
            data=st.session_state["pdf_bytes"],
            file_name="power_quality_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

# ── Raw data download ─────────────────────────────────────
st.markdown("---")
st.subheader("⬇️  Download Raw Data")
data = st.session_state.get("data")
if data is not None:
    st.download_button(
        label="⬇️  Download Waveform CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="waveform_data.csv",
        mime="text/csv",
    )
else:
    st.info("No waveform data available to download.")
