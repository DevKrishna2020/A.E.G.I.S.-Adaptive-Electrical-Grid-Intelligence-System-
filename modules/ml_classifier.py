"""
ml_classifier.py
----------------
Machine-learning pipeline for power quality condition classification.

Uses a Random Forest classifier trained on a synthetic dataset with
physically consistent feature distributions.

IMPORTANT: This model is trained and evaluated on synthetic data only.
Performance on real electrical systems has not been validated.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)


# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------

CLASSES = [
    "Normal",
    "Voltage Sag",
    "Voltage Swell",
    "Voltage Interruption",
    "Harmonic Distortion",
    "Frequency Deviation",
]

FEATURE_NAMES = [
    "vrms",
    "voltage_deviation_pct",
    "irms",
    "frequency",
    "freq_deviation_hz",
    "thd_pct",
    "v_ratio",
    "skewness",
    "kurtosis",
]


# ---------------------------------------------------------------------------
# Synthetic training dataset generation
# ---------------------------------------------------------------------------

def _rng_samples(rng, loc, scale, size, clip_low=None, clip_high=None):
    s = rng.normal(loc, scale, size)
    if clip_low is not None:
        s = np.clip(s, clip_low, None)
    if clip_high is not None:
        s = np.clip(s, None, clip_high)
    return s


def generate_training_dataset(
    n_per_class: int = 500,
    nominal_voltage: float = 230.0,
    nominal_frequency: float = 50.0,
    random_seed: int = 0,
) -> pd.DataFrame:
    """
    Generate a synthetic labelled dataset for classifier training.

    Feature distributions are constructed to be physically consistent with
    each power quality condition.  They do not represent measured data from
    real electrical systems.

    Returns
    -------
    pd.DataFrame with feature columns + 'label' column.
    """
    rng = np.random.default_rng(random_seed)
    rows = []

    def add_class(label, vrms_loc, vrms_scale, irms_scale_factor,
                  freq_loc, freq_scale, thd_loc, thd_scale,
                  skew_loc, kurt_loc):
        for _ in range(n_per_class):
            v = max(0.5, rng.normal(vrms_loc, vrms_scale))
            v_dev = 100.0 * (v - nominal_voltage) / nominal_voltage
            i = max(0.01, v / 10.0 * (1.0 + rng.normal(0, irms_scale_factor)))
            f = max(45.0, min(55.0, rng.normal(freq_loc, freq_scale)))
            f_dev = abs(f - nominal_frequency)
            thd = max(0.0, rng.normal(thd_loc, thd_scale))
            v_ratio = v / nominal_voltage
            skew = rng.normal(skew_loc, 0.3)
            kurt = max(0.0, rng.normal(kurt_loc, 0.5))
            rows.append([v, v_dev, i, f, f_dev, thd, v_ratio, skew, kurt, label])

    # Normal
    add_class("Normal",
              vrms_loc=nominal_voltage, vrms_scale=2.0,
              irms_scale_factor=0.02,
              freq_loc=nominal_frequency, freq_scale=0.05,
              thd_loc=2.0, thd_scale=0.5,
              skew_loc=0.0, kurt_loc=3.0)

    # Voltage Sag  (0.1–0.9 p.u.)
    sag_v = nominal_voltage * rng.uniform(0.5, 0.89, n_per_class)
    for v in sag_v:
        v_dev = 100.0 * (v - nominal_voltage) / nominal_voltage
        i = max(0.01, v / 10.0 * rng.uniform(0.8, 1.2))
        f = rng.normal(nominal_frequency, 0.05)
        f_dev = abs(f - nominal_frequency)
        thd = max(0.0, rng.normal(2.5, 0.8))
        v_ratio = v / nominal_voltage
        skew = rng.normal(-0.2, 0.3)
        kurt = max(0.0, rng.normal(3.5, 0.5))
        rows.append([v, v_dev, i, f, f_dev, thd, v_ratio, skew, kurt, "Voltage Sag"])

    # Voltage Swell  (1.1–1.5 p.u.)
    swell_v = nominal_voltage * rng.uniform(1.1, 1.40, n_per_class)
    for v in swell_v:
        v_dev = 100.0 * (v - nominal_voltage) / nominal_voltage
        i = max(0.01, v / 10.0 * rng.uniform(0.9, 1.1))
        f = rng.normal(nominal_frequency, 0.05)
        f_dev = abs(f - nominal_frequency)
        thd = max(0.0, rng.normal(2.0, 0.6))
        v_ratio = v / nominal_voltage
        skew = rng.normal(0.2, 0.3)
        kurt = max(0.0, rng.normal(3.0, 0.5))
        rows.append([v, v_dev, i, f, f_dev, thd, v_ratio, skew, kurt, "Voltage Swell"])

    # Voltage Interruption  (< 0.1 p.u.)
    int_v = nominal_voltage * rng.uniform(0.0, 0.09, n_per_class)
    for v in int_v:
        v_dev = 100.0 * (v - nominal_voltage) / nominal_voltage
        i = max(0.0, v / 10.0 * rng.uniform(0.0, 0.2))
        f = rng.normal(nominal_frequency, 0.1)
        f_dev = abs(f - nominal_frequency)
        thd = max(0.0, rng.normal(15.0, 5.0))
        v_ratio = v / nominal_voltage
        skew = rng.normal(0.0, 0.8)
        kurt = max(0.0, rng.normal(4.0, 1.0))
        rows.append([v, v_dev, i, f, f_dev, thd, v_ratio, skew, kurt, "Voltage Interruption"])

    # Harmonic Distortion  (normal voltage, high THD)
    add_class("Harmonic Distortion",
              vrms_loc=nominal_voltage, vrms_scale=3.0,
              irms_scale_factor=0.05,
              freq_loc=nominal_frequency, freq_scale=0.05,
              thd_loc=18.0, thd_scale=5.0,
              skew_loc=0.3, kurt_loc=4.5)

    # Frequency Deviation  (normal voltage, off-frequency)
    freq_devs = rng.choice([-1, 1], n_per_class) * rng.uniform(0.5, 2.5, n_per_class)
    for fd in freq_devs:
        v = max(0.5, rng.normal(nominal_voltage, 3.0))
        v_dev = 100.0 * (v - nominal_voltage) / nominal_voltage
        i = max(0.01, v / 10.0 * (1.0 + rng.normal(0, 0.03)))
        f = nominal_frequency + fd
        thd = max(0.0, rng.normal(2.5, 0.8))
        v_ratio = v / nominal_voltage
        skew = rng.normal(0.0, 0.2)
        kurt = max(0.0, rng.normal(3.0, 0.4))
        rows.append([v, v_dev, i, f, abs(fd), thd, v_ratio, skew, kurt, "Frequency Deviation"])

    cols = FEATURE_NAMES + ["label"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Model training and evaluation
# ---------------------------------------------------------------------------

def train_classifier(
    df: pd.DataFrame | None = None,
    n_per_class: int = 500,
    nominal_voltage: float = 230.0,
    nominal_frequency: float = 50.0,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict:
    """
    Train a Random Forest classifier and return evaluation metrics.

    Parameters
    ----------
    df : pd.DataFrame or None
        Labelled dataset with feature columns + 'label'.
        If None, the built-in synthetic dataset is generated.
    n_per_class : int
        Samples per class when using the synthetic dataset.
    test_size : float
        Fraction of data reserved for testing.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        model, label_encoder, feature_names,
        accuracy, precision, recall, f1,
        confusion_matrix, class_names,
        classification_report, train_size, test_size
    """
    if df is None:
        df = generate_training_dataset(
            n_per_class=n_per_class,
            nominal_voltage=nominal_voltage,
            nominal_frequency=nominal_frequency,
            random_seed=random_seed,
        )

    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["label"].to_numpy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_seed, stratify=y_enc
    )

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec   = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1    = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm    = confusion_matrix(y_test, y_pred)
    cr    = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

    return {
        "model":                   clf,
        "label_encoder":           le,
        "feature_names":           FEATURE_NAMES,
        "accuracy":                round(acc, 4),
        "precision":               round(prec, 4),
        "recall":                  round(rec, 4),
        "f1":                      round(f1, 4),
        "confusion_matrix":        cm,
        "class_names":             list(le.classes_),
        "classification_report":   cr,
        "train_size":              len(X_train),
        "test_size":               len(X_test),
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    model,
    label_encoder,
    vrms: float,
    irms: float,
    frequency: float,
    thd_pct: float,
    nominal_voltage: float = 230.0,
    nominal_frequency: float = 50.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> dict:
    """
    Predict power quality condition for a single measurement vector.

    Returns
    -------
    dict with keys: predicted_class, confidence, probabilities
    """
    v_dev = 100.0 * (vrms - nominal_voltage) / nominal_voltage
    f_dev = abs(frequency - nominal_frequency)
    v_ratio = vrms / nominal_voltage

    features = np.array([[
        vrms, v_dev, irms, frequency, f_dev,
        thd_pct, v_ratio, skewness, kurtosis,
    ]])

    proba = model.predict_proba(features)[0]
    pred_enc = int(np.argmax(proba))
    predicted_class = label_encoder.inverse_transform([pred_enc])[0]
    confidence = float(proba[pred_enc])

    class_proba = {
        label_encoder.classes_[i]: round(float(p), 4)
        for i, p in enumerate(proba)
    }

    return {
        "predicted_class": predicted_class,
        "confidence":      round(confidence, 4),
        "probabilities":   class_proba,
    }
