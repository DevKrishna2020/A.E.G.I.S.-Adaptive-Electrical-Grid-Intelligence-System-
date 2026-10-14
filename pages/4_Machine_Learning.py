"""
4_Machine_Learning.py — Machine Learning Analysis page
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from modules.ml_classifier import (
    train_classifier, predict, CLASSES, FEATURE_NAMES, generate_training_dataset,
)
from modules.visualization import plot_confusion_matrix, plot_class_probabilities

st.set_page_config(page_title="Machine Learning", page_icon="🤖", layout="wide")
st.title("🤖 Machine Learning Power Quality Classifier")
st.markdown(
    "A **Random Forest** classifier trained on a synthetic dataset classifies "
    "power quality conditions from extracted features."
)
st.warning(
    "⚠️ **Important:** The model is trained on synthetic data only. "
    "Performance on real electrical systems has not been validated. "
    "Use ML predictions alongside the rule-based detector, not as a sole diagnosis."
)

nom_v = float(st.session_state.get("nominal_voltage",  230.0))
nom_f = float(st.session_state.get("nominal_frequency", 50.0))

# ── Training configuration ────────────────────────────────
with st.expander("⚙️  Training Configuration", expanded=True):
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        n_per_class = st.number_input("Training samples per class", 100, 2000, 500, 100)
    with tc2:
        test_frac = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)
    with tc3:
        seed = st.number_input("Random seed", 0, 9999, 42, 1)

if st.button("🚀 Train / Retrain Model", type="primary"):
    with st.spinner("Training Random Forest classifier..."):
        info = train_classifier(
            n_per_class=int(n_per_class),
            nominal_voltage=nom_v,
            nominal_frequency=nom_f,
            test_size=test_frac,
            random_seed=int(seed),
        )
    st.session_state["ml_model_info"] = info
    st.success(
        f"✅ Model trained on {info['train_size']:,} samples | "
        f"Tested on {info['test_size']:,} samples | "
        f"Accuracy: {info['accuracy']*100:.2f}%"
    )

info = st.session_state.get("ml_model_info")

# Auto-train on first visit
if info is None:
    with st.spinner("Training model with default settings..."):
        info = train_classifier(n_per_class=500, nominal_voltage=nom_v,
                                nominal_frequency=nom_f, test_size=0.2, random_seed=42)
    st.session_state["ml_model_info"] = info

# ── Evaluation metrics ────────────────────────────────────
st.markdown("---")
st.subheader("Model Evaluation Metrics (Synthetic Test Set)")

em1, em2, em3, em4 = st.columns(4)
em1.metric("Accuracy",  f"{info['accuracy']*100:.2f}%")
em2.metric("Precision", f"{info['precision']*100:.2f}%")
em3.metric("Recall",    f"{info['recall']*100:.2f}%")
em4.metric("F1-Score",  f"{info['f1']*100:.2f}%")

col_cm, col_cr = st.columns([1, 1])
with col_cm:
    st.subheader("Confusion Matrix")
    st.plotly_chart(
        plot_confusion_matrix(info["confusion_matrix"], info["class_names"]),
        use_container_width=True,
    )
with col_cr:
    st.subheader("Classification Report")
    st.code(info["classification_report"], language="text")

# ── Feature importances ───────────────────────────────────
st.markdown("---")
st.subheader("Feature Importances")
importances = info["model"].feature_importances_
fi_df = pd.DataFrame({
    "Feature": FEATURE_NAMES,
    "Importance": importances,
}).sort_values("Importance", ascending=False)

import plotly.express as px
fig_fi = px.bar(
    fi_df, x="Feature", y="Importance",
    title="Random Forest Feature Importances",
    color="Importance", color_continuous_scale="Blues",
    template="plotly_white",
)
st.plotly_chart(fig_fi, use_container_width=True)

# ── Predict on current data ───────────────────────────────
st.markdown("---")
st.subheader("Classify Current Dataset")

metrics = st.session_state.get("metrics")
if metrics:
    vrms  = metrics.get("vrms", nom_v)
    irms  = metrics.get("irms", nom_v / 10.0) or nom_v / 10.0
    freq  = metrics.get("frequency", nom_f)
    thd   = metrics.get("thd_pct") or 2.0

    # Compute skewness and kurtosis from waveform if available
    data: pd.DataFrame | None = st.session_state.get("data")
    skew_val, kurt_val = 0.0, 3.0
    if data is not None and "voltage" in data.columns:
        v = data["voltage"].to_numpy(dtype=float)
        from scipy.stats import skew as sp_skew, kurtosis as sp_kurt
        skew_val = float(sp_skew(v))
        kurt_val = float(sp_kurt(v, fisher=False))  # Pearson kurtosis (normal ≈ 3)

    prediction = predict(
        model=info["model"],
        label_encoder=info["label_encoder"],
        vrms=vrms, irms=irms,
        frequency=freq, thd_pct=thd,
        nominal_voltage=nom_v,
        nominal_frequency=nom_f,
        skewness=skew_val,
        kurtosis=kurt_val,
    )
    st.session_state["ml_prediction"] = prediction

    pc1, pc2 = st.columns(2)
    with pc1:
        st.metric("Predicted Condition", prediction["predicted_class"])
        st.metric("Confidence", f"{prediction['confidence']*100:.1f}%")
        st.caption(
            "ℹ️ The ML result is a statistical estimate based on synthetic training data. "
            "Compare with the rule-based detection result."
        )
    with pc2:
        st.subheader("Class Probabilities")
        st.plotly_chart(plot_class_probabilities(prediction["probabilities"]),
                        use_container_width=True)

    # Rule vs ML comparison
    rule_result = st.session_state.get("snapshot_result")
    disturbances = st.session_state.get("disturbances")
    if rule_result:
        rule_status = rule_result.get("status", "N/A")
    elif disturbances is not None and len(disturbances) > 0:
        rule_status = disturbances["disturbance_type"].iloc[0]
    else:
        rule_status = "Normal (no rule result)"

    st.markdown("---")
    st.subheader("Rule-Based vs ML Comparison")
    comp1, comp2 = st.columns(2)
    comp1.metric("Rule-Based Result",    rule_status)
    comp2.metric("ML Predicted Class",   prediction["predicted_class"])

    if rule_status.lower().replace(" ", "_") != prediction["predicted_class"].lower().replace(" ", "_"):
        st.info(
            "ℹ️ The rule-based and ML detectors disagree. This can occur when the "
            "waveform contains multiple overlapping conditions, borderline parameter "
            "values, or when the synthetic training distribution does not fully capture "
            "the measured scenario. Manual inspection is recommended."
        )
    else:
        st.success("✅ Both detectors agree on the condition.")

else:
    st.info("Run **Electrical Analysis** first so the ML module has features to classify.")

# ── Synthetic dataset preview ─────────────────────────────
st.markdown("---")
with st.expander("📋  Preview Synthetic Training Dataset (first 50 rows)"):
    st.caption(
        "This dataset is generated programmatically. Feature distributions are "
        "designed to be physically consistent but do not represent measured data."
    )
    preview = generate_training_dataset(n_per_class=50, nominal_voltage=nom_v,
                                        nominal_frequency=nom_f, random_seed=0)
    st.dataframe(preview.head(50), use_container_width=True)
