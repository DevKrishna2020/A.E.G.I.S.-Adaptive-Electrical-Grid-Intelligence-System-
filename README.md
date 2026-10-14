# AI-Based Power Quality Monitoring System

A complete, functional Python/Streamlit web application that monitors electrical parameters,
identifies power quality disturbances, classifies conditions using machine learning,
and generates downloadable reports.

---

## Features

| Feature | Description |
|---|---|
| **Synthetic Data Generation** | Simulate normal, sag, swell, interruption, harmonic, frequency-deviation, and multiple-disturbance scenarios |
| **CSV Upload** | Upload your own electrical measurement dataset |
| **Manual Input** | Enter RMS summary values for instant rule-based assessment |
| **Electrical Calculations** | RMS voltage & current, frequency estimation (FFT), THD, active/reactive/apparent power, power factor |
| **Rule-Based Detection** | Configurable thresholds per IEC 61000 / EN 50160; rolling-RMS time-series event detection |
| **Machine Learning** | Random Forest classifier trained on synthetic data; confusion matrix and per-class metrics |
| **Interactive Visualizations** | Plotly waveforms, RMS trend, frequency variation, harmonic spectrum, disturbance timeline, waveform comparison, confusion matrix |
| **Reports** | Downloadable PDF (fpdf2) and CSV reports with metrics, events, ML results, and recommendations |
| **Recommendations Engine** | Condition-specific educational guidance with professional disclaimer |

---

## Project Structure

```
power_quality_monitor/
├── app.py                          ← Home Dashboard (Streamlit entry point)
├── requirements.txt
├── README.md
├── data/
│   ├── sample_power_quality.csv    ← Sample voltage-sag dataset
│   └── generate_sample_csv.py      ← Script to regenerate sample data
├── modules/
│   ├── __init__.py
│   ├── data_loader.py              ← Synthetic data generation + CSV loading
│   ├── signal_processing.py        ← RMS, FFT, frequency estimation, THD, rolling RMS
│   ├── electrical_metrics.py       ← Full waveform and manual-input parameter calculation
│   ├── disturbance_detection.py    ← Rule-based snapshot and time-series detection
│   ├── ml_classifier.py            ← Random Forest training, evaluation, prediction
│   ├── visualization.py            ← Plotly chart builders
│   ├── report_generator.py         ← PDF (fpdf2) and CSV report generation
│   └── recommendations.py          ← Condition-specific recommendations engine
├── pages/
│   ├── 1_Data_Input.py             ← Simulation / CSV upload / manual input
│   ├── 2_Electrical_Analysis.py    ← Parameter calculations and harmonic table
│   ├── 3_Disturbance_Detection.py  ← Threshold configuration and event detection
│   ├── 4_Machine_Learning.py       ← ML training, evaluation, and prediction
│   ├── 5_Visualizations.py         ← All interactive charts
│   ├── 6_Reports.py                ← Recommendations, downloads (PDF + CSV)
│   └── 7_About.py                  ← Project documentation and architecture
└── tests/
    ├── test_electrical_metrics.py  ← 27 unit tests for calculations
    └── test_disturbance_detection.py ← 19 unit tests for detection logic
```

---

## Setup and Running on Windows

### 1. Prerequisites

- Python 3.11 or later: https://www.python.org/downloads/
- pip (bundled with Python)

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the application

```powershell
python -m streamlit run app.py
```

Streamlit will open the application automatically in your browser at `http://localhost:8501`.

### 5. Run the automated tests

```powershell
python -m pytest tests/ -v
```

---

## Demonstration Guide (College Presentation)

### Feature 1 — Synthetic Data and Waveforms
1. Open **Data Input** in the sidebar.
2. Select scenario **"Voltage Sag"**, set severity to 0.7, click **Generate Electrical Data**.
3. Go to **Electrical Analysis** — show RMS voltage is below nominal, THD table, power metrics.

### Feature 2 — Rule-Based Disturbance Detection
1. Go to **Disturbance Detection**.
2. Expand thresholds panel — explain sag/swell/interruption criteria.
3. Click **Run Time-Series Detection** — show the events table and disturbance timeline.
4. Point to the rolling RMS chart and explain the threshold lines.

### Feature 3 — Machine Learning Classification
1. Go to **Machine Learning**.
2. Click **Train / Retrain Model** — show accuracy, confusion matrix, classification report.
3. Observe the **Predicted Condition** for the loaded dataset.
4. Show the **Feature Importances** chart and explain which features matter most.
5. Compare ML result with rule-based result in the comparison section.

### Feature 4 — Interactive Visualizations
1. Go to **Visualizations** — demonstrate zoom/pan on the voltage waveform.
2. Show the harmonic spectrum bar chart (use the "Harmonic Distortion" scenario).
3. Change the comparison scenario dropdown and show the side-by-side waveform comparison.

### Feature 5 — Reports and Downloads
1. Go to **Reports**.
2. Show the recommendations section and explain the disclaimer.
3. Download the CSV report — open it in Excel to show the structured content.
4. Click **Generate PDF Report** and download.

### Feature 6 — CSV Upload
1. Download the sample CSV from **Data Input → Option B**.
2. Re-upload the same file to demonstrate CSV loading and validation.

---

## Limitations

- Trained and tested on **synthetic data only**. Results should not be used for real
  electrical engineering decisions without independent verification.
- Single-phase analysis only.
- THD accuracy depends on waveform duration and sampling rate.
- Not a substitute for professional power quality monitoring equipment or analysis.

---

## Technology Stack

Python · Streamlit · Pandas · NumPy · SciPy · Scikit-learn · Plotly · fpdf2

---

## Disclaimer

This application is designed for **educational and demonstration purposes only**.
All analysis results must be reviewed and verified by a qualified electrical engineer
before any equipment, wiring, or protection settings are changed.
