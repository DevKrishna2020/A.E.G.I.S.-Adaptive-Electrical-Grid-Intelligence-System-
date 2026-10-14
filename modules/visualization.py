"""
visualization.py
----------------
Plotly-based chart builders for the power quality monitoring application.
All functions return plotly.graph_objects.Figure objects ready for st.plotly_chart().
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Voltage waveform
# ---------------------------------------------------------------------------

def plot_voltage_waveform(df: pd.DataFrame, title: str = "Voltage Waveform") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["voltage"],
        mode="lines", name="Voltage (V)",
        line=dict(color="#2563EB", width=1),
        hovertemplate="t=%{x:.4f} s<br>V=%{y:.2f} V<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Voltage (V)",
        template="plotly_white",
        height=350,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# ---------------------------------------------------------------------------
# Current waveform
# ---------------------------------------------------------------------------

def plot_current_waveform(df: pd.DataFrame, title: str = "Current Waveform") -> go.Figure:
    if "current" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No current data available", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=16))
        fig.update_layout(title=title, template="plotly_white", height=300)
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["current"],
        mode="lines", name="Current (A)",
        line=dict(color="#DC2626", width=1),
        hovertemplate="t=%{x:.4f} s<br>I=%{y:.3f} A<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Current (A)",
        template="plotly_white",
        height=350,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# ---------------------------------------------------------------------------
# RMS voltage trend
# ---------------------------------------------------------------------------

def plot_rms_trend(
    timestamps: np.ndarray,
    vrms_rolling: np.ndarray,
    nominal_voltage: float = 230.0,
    thresholds: dict | None = None,
) -> go.Figure:
    thr = {"sag_threshold": 0.90, "swell_threshold": 1.10, **(thresholds or {})}
    v_sag  = nominal_voltage * thr["sag_threshold"]
    v_swell = nominal_voltage * thr["swell_threshold"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps, y=vrms_rolling,
        mode="lines", name="Rolling RMS Voltage",
        line=dict(color="#2563EB", width=1.5),
        hovertemplate="t=%{x:.4f} s<br>Vrms=%{y:.2f} V<extra></extra>",
    ))
    fig.add_hline(y=nominal_voltage, line_dash="dash", line_color="green",
                  annotation_text=f"Nominal {nominal_voltage} V")
    fig.add_hline(y=v_sag, line_dash="dot", line_color="orange",
                  annotation_text=f"Sag threshold {v_sag:.1f} V")
    fig.add_hline(y=v_swell, line_dash="dot", line_color="red",
                  annotation_text=f"Swell threshold {v_swell:.1f} V")
    fig.update_layout(
        title="RMS Voltage Trend",
        xaxis_title="Time (s)",
        yaxis_title="RMS Voltage (V)",
        template="plotly_white",
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Frequency variation
# ---------------------------------------------------------------------------

def plot_frequency_trend(df: pd.DataFrame, nominal_frequency: float = 50.0) -> go.Figure:
    fig = go.Figure()
    if "frequency" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["frequency"],
            mode="lines", name="Frequency (Hz)",
            line=dict(color="#7C3AED", width=1.5),
            hovertemplate="t=%{x:.4f} s<br>f=%{y:.3f} Hz<extra></extra>",
        ))
    fig.add_hline(y=nominal_frequency, line_dash="dash", line_color="green",
                  annotation_text=f"Nominal {nominal_frequency} Hz")
    fig.add_hline(y=nominal_frequency + 0.5, line_dash="dot", line_color="orange",
                  annotation_text="+0.5 Hz limit")
    fig.add_hline(y=nominal_frequency - 0.5, line_dash="dot", line_color="orange",
                  annotation_text="-0.5 Hz limit")
    fig.update_layout(
        title="Frequency Variation",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        template="plotly_white",
        height=350,
    )
    return fig


# ---------------------------------------------------------------------------
# Harmonic spectrum / THD bar chart
# ---------------------------------------------------------------------------

def plot_harmonic_spectrum(harmonics: dict, f1: float = 50.0) -> go.Figure:
    if not harmonics:
        fig = go.Figure()
        fig.add_annotation(text="No harmonic data available", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=14))
        fig.update_layout(title="Harmonic Spectrum", template="plotly_white", height=300)
        return fig

    orders = list(harmonics.keys())
    freqs  = [h * f1 for h in orders]
    amps   = [harmonics[h] for h in orders]

    colors = ["#2563EB" if h == 1 else "#DC2626" for h in orders]
    fig = go.Figure(go.Bar(
        x=[f"H{h} ({freqs[i]:.0f} Hz)" for i, h in enumerate(orders)],
        y=amps,
        marker_color=colors,
        hovertemplate="Order %{x}<br>RMS=%{y:.4f} V<extra></extra>",
    ))
    fig.update_layout(
        title="Harmonic Spectrum (Voltage)",
        xaxis_title="Harmonic Order",
        yaxis_title="RMS Amplitude (V)",
        template="plotly_white",
        height=350,
    )
    return fig


# ---------------------------------------------------------------------------
# Disturbance timeline
# ---------------------------------------------------------------------------

def plot_disturbance_timeline(events_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if events_df is None or len(events_df) == 0:
        fig.add_annotation(text="No disturbances detected", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5, font=dict(size=16))
        fig.update_layout(title="Disturbance Timeline", template="plotly_white", height=250)
        return fig

    color_map = {
        "Voltage Sag":          "#F59E0B",
        "Voltage Swell":        "#EF4444",
        "Voltage Interruption": "#7F1D1D",
        "Harmonic Distortion":  "#7C3AED",
        "Frequency Deviation":  "#2563EB",
    }

    for _, row in events_df.iterrows():
        color = color_map.get(row["disturbance_type"], "#6B7280")
        fig.add_shape(
            type="rect",
            x0=row["start_time"], x1=row["end_time"],
            y0=0, y1=1,
            fillcolor=color, opacity=0.5,
            line_width=0,
            layer="below",
        )
        fig.add_annotation(
            x=(row["start_time"] + row["end_time"]) / 2,
            y=0.5,
            text=row["disturbance_type"],
            showarrow=False,
            font=dict(size=10, color="white"),
            bgcolor=color,
        )

    fig.update_layout(
        title="Power Quality Disturbance Timeline",
        xaxis_title="Time (s)",
        yaxis=dict(showticklabels=False, range=[0, 1]),
        template="plotly_white",
        height=200,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Waveform comparison: normal vs disturbed
# ---------------------------------------------------------------------------

def plot_waveform_comparison(df_normal: pd.DataFrame, df_disturbed: pd.DataFrame,
                              label: str = "Disturbed") -> go.Figure:
    fig = go.Figure()
    n = min(len(df_normal), len(df_disturbed), 1000)
    t = df_normal["timestamp"].iloc[:n].to_numpy()

    fig.add_trace(go.Scatter(
        x=t, y=df_normal["voltage"].iloc[:n].to_numpy(),
        mode="lines", name="Normal",
        line=dict(color="#22C55E", width=1.2),
    ))
    fig.add_trace(go.Scatter(
        x=t, y=df_disturbed["voltage"].iloc[:n].to_numpy(),
        mode="lines", name=label,
        line=dict(color="#EF4444", width=1.2, dash="dash"),
    ))
    fig.update_layout(
        title="Waveform Comparison: Normal vs Disturbed",
        xaxis_title="Time (s)",
        yaxis_title="Voltage (V)",
        template="plotly_white",
        height=370,
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> go.Figure:
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names,
        y=class_names,
        color_continuous_scale="Blues",
        text_auto=True,
        aspect="auto",
        title="ML Classifier — Confusion Matrix",
    )
    fig.update_layout(template="plotly_white", height=500)
    return fig


# ---------------------------------------------------------------------------
# ML class probability bar chart
# ---------------------------------------------------------------------------

def plot_class_probabilities(probabilities: dict) -> go.Figure:
    classes = list(probabilities.keys())
    probs   = [probabilities[c] * 100 for c in classes]
    colors  = ["#2563EB" if p == max(probs) else "#93C5FD" for p in probs]
    fig = go.Figure(go.Bar(
        x=classes, y=probs,
        marker_color=colors,
        hovertemplate="%{x}<br>%{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="ML Prediction — Class Probabilities",
        xaxis_title="Condition",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 100],
        template="plotly_white",
        height=350,
    )
    return fig
