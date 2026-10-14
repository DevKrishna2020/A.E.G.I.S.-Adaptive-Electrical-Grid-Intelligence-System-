"""
recommendations.py
------------------
Rule-based recommendation engine for detected power quality conditions.
"""

RECOMMENDATIONS: dict[str, dict] = {
    "Normal": {
        "summary": "No power quality issues detected.",
        "actions": [
            "Continue regular monitoring.",
            "Verify calibration of measurement equipment periodically.",
        ],
        "severity_level": "info",
    },
    "Voltage Sag": {
        "summary": "A voltage sag (dip) has been detected. Voltage dropped below the normal operating band.",
        "actions": [
            "Investigate recent load-switching events (motor starts, transformer energisation).",
            "Check for supply network disturbances or fault-clearing events upstream.",
            "Examine whether conductor voltage drop is contributing to the sag.",
            "Consider voltage support equipment (UPS, voltage stabiliser) for sensitive loads.",
            "Review protection coordination to determine whether a fault caused the sag.",
        ],
        "severity_level": "warning",
    },
    "Voltage Swell": {
        "summary": "A voltage swell has been detected. Voltage exceeded the normal operating band.",
        "actions": [
            "Investigate sudden load disconnection events that may cause voltage rise.",
            "Check voltage regulator and transformer tap settings.",
            "On single-phase or split-phase systems, check for neutral connection issues.",
            "Verify that capacitor bank switching is not causing over-voltage.",
            "Review over-voltage protection settings.",
        ],
        "severity_level": "warning",
    },
    "Voltage Interruption": {
        "summary": "A voltage interruption has been detected. Voltage fell to near zero.",
        "actions": [
            "Investigate supply outage events (upstream fault, protection operation).",
            "Check cable and connector integrity at the supply connection point.",
            "Review protection relay settings and fuse ratings.",
            "Consider uninterruptible power supply (UPS) or backup generation for critical loads.",
            "Coordinate with the distribution network operator if the supply is from the grid.",
        ],
        "severity_level": "critical",
    },
    "Harmonic Distortion": {
        "summary": "Elevated harmonic distortion has been detected in the voltage waveform.",
        "actions": [
            "Identify non-linear loads (variable speed drives, switch-mode power supplies, EV chargers).",
            "Measure current harmonics to determine which loads are contributing.",
            "Evaluate whether harmonic filters (passive or active) are required.",
            "Check transformer ratings — harmonic currents increase losses and temperature.",
            "Review neutral conductor sizing (triplen harmonics add in the neutral).",
        ],
        "severity_level": "warning",
    },
    "Harmonic Distortion (Elevated)": {
        "summary": "THD is elevated above the warning threshold but below the high threshold.",
        "actions": [
            "Monitor harmonic levels over time.",
            "Identify non-linear loads that may be contributing.",
            "Consider further investigation if levels continue to rise.",
        ],
        "severity_level": "info",
    },
    "Harmonic Distortion (High)": {
        "summary": "THD has exceeded the high-distortion threshold. Mitigation is recommended.",
        "actions": [
            "Identify and meter all non-linear loads on the circuit.",
            "Consider active or passive harmonic mitigation.",
            "Verify transformer K-rating suitability.",
            "Check neutral conductor temperatures.",
        ],
        "severity_level": "warning",
    },
    "Frequency Deviation": {
        "summary": "Supply frequency has deviated from the nominal value.",
        "actions": [
            "If connected to the public grid, deviation may indicate a system event. Contact the network operator.",
            "For isolated generators: check governor/speed controller settings.",
            "Verify measurement equipment accuracy (clock, ADC sampling reference).",
            "Review load balance on generation systems.",
            "Frequency-sensitive equipment (motors, clocks, protection relays) may be affected.",
        ],
        "severity_level": "warning",
    },
    "Multiple": {
        "summary": "Multiple power quality disturbances have been detected simultaneously.",
        "actions": [
            "Address each disturbance type individually using the guidance above.",
            "Prioritise voltage interruptions and high-severity sags.",
            "Consider a comprehensive power quality audit.",
        ],
        "severity_level": "critical",
    },
}

DISCLAIMER = (
    "⚠️  Disclaimer: These recommendations are educational and indicative only. "
    "They are generated automatically from rule-based analysis of simulated or "
    "uploaded data. They do not constitute a professional electrical engineering "
    "assessment. All recommendations must be reviewed and verified by a qualified "
    "electrical engineer before any equipment, wiring, or protection settings are "
    "modified."
)


def get_recommendations(disturbances: list[str]) -> list[dict]:
    """
    Return a list of recommendation dicts for the detected disturbances.

    Each dict has keys: disturbance, summary, actions, severity_level.
    """
    if not disturbances:
        rec = RECOMMENDATIONS["Normal"].copy()
        rec["disturbance"] = "Normal"
        return [rec]

    results = []
    for d in disturbances:
        # Try exact match, then partial match
        rec = RECOMMENDATIONS.get(d)
        if rec is None:
            for key in RECOMMENDATIONS:
                if key.lower() in d.lower() or d.lower() in key.lower():
                    rec = RECOMMENDATIONS[key]
                    break
        if rec is None:
            rec = {
                "summary": f"Disturbance detected: {d}.",
                "actions": ["Consult a qualified electrical engineer for assessment."],
                "severity_level": "warning",
            }
        entry = rec.copy()
        entry["disturbance"] = d
        results.append(entry)
    return results
