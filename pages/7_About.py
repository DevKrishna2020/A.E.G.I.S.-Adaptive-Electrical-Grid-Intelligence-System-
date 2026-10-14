"""
7_About.py — About the Project page
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(page_title="About the Project", page_icon="ℹ️", layout="wide")
st.title("ℹ️ About the Project")

st.markdown("""
## AI-Based Power Quality Monitoring System

### Project Introduction

This application demonstrates how Python-based signal processing and machine learning
can be applied to the analysis of electrical power quality. It is designed as a
college project for Electrical and Electronics Engineering students.

---

### Problem Statement

Poor power quality causes equipment failures, production downtime, increased energy
costs, and can violate regulatory requirements. Monitoring and identifying power
quality disturbances in real time is challenging without specialised equipment.

This project addresses the educational gap by providing a functional software tool
that simulates and analyses power quality conditions, helping students understand the
principles of electrical measurement, signal processing, and AI-based classification.

---

### Objectives

1. Simulate and process electrical waveform data for common power quality conditions.
2. Calculate electrical parameters (RMS voltage, current, frequency, THD, power factor).
3. Detect and classify power quality disturbances using rule-based engineering thresholds.
4. Apply machine learning to classify power quality conditions from extracted features.
5. Visualise electrical data interactively for analysis and presentation.
6. Generate downloadable reports suitable for documentation and viva demonstrations.

---

### Methodology

| Step | Description |
|------|-------------|
| **1. Data Acquisition** | Synthetic waveform generation, CSV upload, or manual input |
| **2. Signal Processing** | FFT, RMS, frequency estimation, harmonic analysis |
| **3. Parameter Calculation** | Vrms, Irms, frequency, THD, P, Q, S, PF |
| **4. Rule-Based Detection** | Threshold comparison per IEC 61000 / EN 50160 |
| **5. ML Classification** | Random Forest trained on synthetic feature vectors |
| **6. Visualisation** | Interactive Plotly charts |
| **7. Report Generation** | PDF and CSV downloads |

---

### System Architecture

```
Electrical Data (Simulated / CSV / Manual)
            │
            ▼
    ┌─────────────────────────────────┐
    │       Data Validation           │
    │  (column checks, NaN removal)   │
    └─────────────┬───────────────────┘
                  │
                  ▼
    ┌─────────────────────────────────┐
    │       Signal Processing         │
    │  RMS · FFT · THD · Frequency    │
    └─────────────┬───────────────────┘
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
  Rule-Based          ML Classification
  Detection           (Random Forest)
       │                     │
       └──────────┬──────────┘
                  │
                  ▼
         Recommendations Engine
                  │
                  ▼
         Visualizations & Reports
```

---

### Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.11+** | Primary programming language |
| **Streamlit** | Multi-page web application framework |
| **Pandas** | Data loading, manipulation, CSV I/O |
| **NumPy** | Numerical array operations |
| **SciPy** | FFT, windowing, signal processing |
| **Scikit-learn** | Random Forest classifier, metrics |
| **Plotly** | Interactive charts and graphs |
| **fpdf2** | PDF report generation |

---

### Expected Outcomes

- Students can simulate all major power quality disturbances and observe their waveform effects.
- The rule-based detector provides transparent, configurable classification.
- The ML classifier demonstrates AI application in power systems engineering.
- Downloadable reports support project documentation and viva presentations.

---

### Limitations

1. **Synthetic data only:** The ML model is trained on programmatically generated data.
   Its accuracy on real electrical measurements has not been verified.
2. **Single-phase only:** The application models single-phase systems. Three-phase
   analysis is not implemented.
3. **Short waveforms:** THD accuracy improves with longer waveforms and higher sampling rates.
4. **No hardware interface:** This is a purely software simulation; no physical sensors are connected.
5. **Not a substitute for professional analysis:** Results must be verified by a
   qualified electrical engineer before any action is taken.

---

### Future Enhancements

- Integration with real-time hardware (ADC / DAQ boards) via Python serial or USB interfaces.
- Three-phase power quality analysis (unbalance, sequence components).
- Real-time streaming dashboard using WebSocket or MQTT.
- Deep learning models (LSTM, CNN) for waveform-level event detection.
- Cloud deployment with historical data storage and trend analysis.
- IEEE 1159 / IEC 61000-4-30 compliant event classification with duration criteria.

---

### Overall Workflow

```
Electrical Data
      │
      ▼
Data Validation ──→ Signal Processing ──→ Electrical Parameter Calculation
                                                        │
                              ┌─────────────────────────┤
                              │                         │
                              ▼                         ▼
                    Rule-Based Detection      ML Classification
                              │                         │
                              └──────────┬──────────────┘
                                         │
                                         ▼
                                   Visualization
                                         │
                                         ▼
                                 Report Generation
```

---
""")

st.info(
    "This project was built using Python and Streamlit as part of an "
    "AI/ML-based electrical engineering college project. "
    "It demonstrates the application of signal processing and machine learning "
    "to power quality analysis."
)
