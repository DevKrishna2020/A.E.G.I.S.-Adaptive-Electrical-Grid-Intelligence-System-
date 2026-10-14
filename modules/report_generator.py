"""
report_generator.py
-------------------
Generate PDF and CSV reports from analysis results.
Uses fpdf2 for PDF generation.
"""

import io
import csv
from datetime import datetime
from fpdf import FPDF


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------

def generate_csv_report(metrics: dict, disturbances_df, recommendations: list) -> bytes:
    """
    Build a CSV report containing metrics, disturbances, and recommendations.

    Returns
    -------
    bytes : UTF-8 encoded CSV content.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow(["Power Quality Analysis Report"])
    writer.writerow(["Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow([])

    # Metrics
    writer.writerow(["=== Electrical Metrics ==="])
    writer.writerow(["Parameter", "Value", "Unit"])
    metric_rows = [
        ("RMS Voltage",      metrics.get("vrms"),         "V"),
        ("RMS Current",      metrics.get("irms"),         "A"),
        ("Frequency",        metrics.get("frequency"),    "Hz"),
        ("Power Factor",     metrics.get("power_factor"), ""),
        ("Active Power",     metrics.get("active_power"), "W"),
        ("Apparent Power",   metrics.get("apparent_power"), "VA"),
        ("Reactive Power",   metrics.get("reactive_power"), "VAR"),
        ("Voltage THD",      metrics.get("thd_pct"),      "%"),
    ]
    for name, val, unit in metric_rows:
        writer.writerow([name, val if val is not None else "N/A", unit])

    writer.writerow([])

    # Disturbances
    writer.writerow(["=== Detected Disturbances ==="])
    if disturbances_df is not None and len(disturbances_df) > 0:
        writer.writerow(list(disturbances_df.columns))
        for _, row in disturbances_df.iterrows():
            writer.writerow(list(row))
    else:
        writer.writerow(["No disturbances detected."])

    writer.writerow([])

    # Recommendations
    writer.writerow(["=== Recommendations ==="])
    for rec in recommendations:
        writer.writerow([rec.get("disturbance", ""), rec.get("summary", "")])
        for action in rec.get("actions", []):
            writer.writerow(["", f"• {action}"])

    return buf.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

class _PQReport(FPDF):
    """Custom FPDF subclass with header and footer."""

    def header(self):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(37, 99, 235)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, "AI-Based Power Quality Monitoring System", ln=True, align="C", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()} | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")


def _section_title(pdf: _PQReport, title: str):
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(239, 246, 255)
    pdf.cell(0, 8, title, ln=True, fill=True)
    pdf.ln(1)


def _row(pdf: _PQReport, label: str, value: str):
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(70, 7, label + ":", border=0)
    pdf.cell(0, 7, str(value), border=0, ln=True)


def generate_pdf_report(
    metrics: dict,
    disturbances_df,
    recommendations: list,
    ml_results: dict | None,
    dataset_description: str = "Simulated / Uploaded Dataset",
    nominal_voltage: float = 230.0,
    nominal_frequency: float = 50.0,
) -> bytes:
    """
    Generate a PDF report and return raw bytes.
    """
    pdf = _PQReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    # Title block
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Power Quality Analysis Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(4)

    # Dataset info
    _section_title(pdf, "1. Dataset Description")
    _row(pdf, "Dataset", dataset_description)
    _row(pdf, "Nominal Voltage", f"{nominal_voltage} V RMS")
    _row(pdf, "Nominal Frequency", f"{nominal_frequency} Hz")
    pdf.ln(3)

    # Electrical metrics
    _section_title(pdf, "2. Calculated Electrical Metrics")
    metric_rows = [
        ("RMS Voltage",      metrics.get("vrms"),          "V"),
        ("RMS Current",      metrics.get("irms"),          "A"),
        ("Frequency",        metrics.get("frequency"),     "Hz"),
        ("Power Factor",     metrics.get("power_factor"),  ""),
        ("Active Power",     metrics.get("active_power"),  "W"),
        ("Apparent Power",   metrics.get("apparent_power"),"VA"),
        ("Reactive Power",   metrics.get("reactive_power"),"VAR"),
        ("Voltage THD",      metrics.get("thd_pct"),       "%"),
    ]
    for label, val, unit in metric_rows:
        display = f"{val} {unit}".strip() if val is not None else "N/A"
        _row(pdf, label, display)
    pdf.ln(3)

    # Disturbances
    _section_title(pdf, "3. Detected Disturbances")
    if disturbances_df is not None and len(disturbances_df) > 0:
        for _, row in disturbances_df.iterrows():
            line = (f"{row.get('disturbance_type','?')}  |  "
                    f"Start: {row.get('start_time','?')} s  |  "
                    f"End: {row.get('end_time','?')} s  |  "
                    f"Duration: {row.get('duration_s','?')} s  |  "
                    f"Severity: {row.get('severity','?')}")
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 6, line)
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, "No disturbances detected.", ln=True)
    pdf.ln(3)

    # ML results
    _section_title(pdf, "4. Machine Learning Classification")
    if ml_results:
        _row(pdf, "Predicted Condition", ml_results.get("predicted_class", "N/A"))
        _row(pdf, "Confidence",          f"{ml_results.get('confidence', 0)*100:.1f}%")
        pdf.ln(1)
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5,
            "Note: The ML model was trained on synthetic data only. "
            "Predictions have not been validated against real electrical measurements."
        )
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, "ML analysis not performed.", ln=True)
    pdf.ln(3)

    # ML evaluation metrics
    if ml_results and "accuracy" in ml_results:
        _section_title(pdf, "5. Model Evaluation Metrics (Synthetic Test Set)")
        _row(pdf, "Accuracy",  f"{ml_results['accuracy']*100:.2f}%")
        _row(pdf, "Precision", f"{ml_results['precision']*100:.2f}%")
        _row(pdf, "Recall",    f"{ml_results['recall']*100:.2f}%")
        _row(pdf, "F1-Score",  f"{ml_results['f1']*100:.2f}%")
        pdf.ln(3)

    # Recommendations
    _section_title(pdf, "6. Recommendations")
    for rec in recommendations:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 7, rec.get("disturbance", "General"), ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, rec.get("summary", ""))
        for action in rec.get("actions", []):
            pdf.cell(5)
            pdf.multi_cell(0, 5, f"• {action}")
        pdf.ln(2)

    # Disclaimer
    pdf.ln(2)
    _section_title(pdf, "7. Limitations and Disclaimer")
    pdf.set_font("Helvetica", "I", 9)
    disclaimer = (
        "This analysis was performed on simulated or user-uploaded data using "
        "automated signal processing and a machine learning model trained on "
        "synthetic examples. Results are intended for educational purposes only. "
        "They should not be used as the sole basis for any electrical engineering "
        "decision, equipment modification, or protection setting change. All findings "
        "must be reviewed and verified by a qualified electrical engineer."
    )
    pdf.multi_cell(0, 5, disclaimer)

    return bytes(pdf.output())
