"""Evaluation utilities: metrics, curves, calibration, confidence bins, bootstrap."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from .loader import load_class_names, load_features, load_model, load_scaler
from .preprocess import apply_scaler, preprocess_batch

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "metrics.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def load_dataset(csv_path: Path = Path("thyroidDF.csv")) -> Optional[pd.DataFrame]:
    if not csv_path.exists():
        logging.error("Dataset missing: %s", csv_path)
        return None
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        logging.exception("Failed reading dataset: %s", e)
        return None


def split_data(
    df: pd.DataFrame,
    label_col: str,
    feature_names: List[str],
    test_size: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' missing in dataset.")

    class_names = sorted(df[label_col].dropna().unique().tolist())
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    y = df[label_col].map(label_to_idx).values

    X = preprocess_batch(df, feature_names)
    
    # Check if stratification is possible (need at least 2 samples per class)
    stratify_y = None
    if stratify:
        unique, counts = np.unique(y, return_counts=True)
        if np.all(counts >= 2):
            stratify_y = y
        else:
            logging.warning("Stratification disabled: some classes have <2 samples")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify_y
    )
    return X_train, X_test, y_train, y_test, class_names


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    preds = model.predict(X)
    classes = np.unique(preds)
    n_classes = len(classes)
    proba = np.full((len(preds), n_classes), 1 / max(n_classes, 1))
    for i, p in enumerate(preds):
        if p in classes:
            idx = int(np.where(classes == p)[0][0])
            proba[i, idx] = 0.9
    return proba


def _confidence_bins(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    max_prob = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    for i in range(n_bins):
        mask = (max_prob >= bins[i]) & (max_prob < bins[i + 1] + 1e-8)
        if mask.sum() == 0:
            rows.append({"bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}", "count": 0, "tpr": np.nan})
            continue
        correct = (preds[mask] == y_true[mask]).mean()
        rows.append({"bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}", "count": int(mask.sum()), "tpr": correct})
    return pd.DataFrame(rows)


def bootstrap_ci(metric_fn, y_true, y_pred, n_bootstrap=200, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    stats = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    lower = np.percentile(stats, alpha / 2 * 100)
    upper = np.percentile(stats, (1 - alpha / 2) * 100)
    return float(lower), float(upper)


def evaluate_tabular(
    df: pd.DataFrame,
    label_col: str,
    test_size: float,
    seed: int,
    thresholds: Optional[Dict[int, float]] = None,
    bootstrap: bool = False,
    n_bootstrap: int = 200,
    stratify: bool = True,
) -> Dict:
    model_data = load_model()
    if model_data is None:
        raise FileNotFoundError("model.pkl not found in project root.")
    model = model_data["model"] if isinstance(model_data, dict) else model_data
    scaler = load_scaler()
    feature_names = load_features()

    X_train, X_test, y_train, y_test, class_names = split_data(df, label_col, feature_names, test_size, seed, stratify=stratify)
    X_train = apply_scaler(X_train, scaler)
    X_test = apply_scaler(X_test, scaler)

    # Fit if not already fitted (assume pre-trained; only fit if needed)
    try:
        _ = model.predict(X_test.head(1))
    except Exception:
        model.fit(X_train, y_train)

    probs = predict_proba(model, X_test)
    
    # Ensure probs has correct shape matching class_names
    if probs.shape[1] != len(class_names):
        # Pad or trim probabilities to match class_names
        n_classes = len(class_names)
        if probs.shape[1] < n_classes:
            # Pad with zeros
            padded = np.zeros((probs.shape[0], n_classes))
            padded[:, :probs.shape[1]] = probs
            probs = padded
        else:
            # Trim to match
            probs = probs[:, :n_classes]
        # Renormalize
        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-10)
    
    if thresholds:
        adjusted = []
        for row in probs:
            row_adj = row.copy()
            for idx, th in thresholds.items():
                if idx < len(row_adj) and row_adj[idx] >= th:
                    row_adj[idx] = max(row_adj[idx], th)
            adjusted.append(row_adj / row_adj.sum())
        probs = np.vstack(adjusted)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    
    # Only include classes present in test set
    present_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
    present_class_names = [class_names[i] for i in present_classes if i < len(class_names)]
    report = classification_report(
        y_test, y_pred, labels=present_classes, target_names=present_class_names, output_dict=True, zero_division=0
    )

    # ROC / PR
    y_true_bin = label_binarize(y_test, classes=list(range(len(class_names))))
    roc_data = []
    pr_data = []
    for i, cls in enumerate(class_names):
        try:
            if i >= probs.shape[1]:
                continue
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
            prec_c, rec_c, _ = precision_recall_curve(y_true_bin[:, i], probs[:, i])
            try:
                roc_auc = roc_auc_score(y_true_bin[:, i], probs[:, i])
            except ValueError:
                roc_auc = np.nan
            pr_auc = np.trapz(prec_c, rec_c)
            roc_data.append({"class": cls, "fpr": fpr, "tpr": tpr, "auc": roc_auc})
            pr_data.append({"class": cls, "precision": prec_c, "recall": rec_c, "pr_auc": pr_auc})
        except Exception as e:
            logging.warning(f"Skipping ROC/PR for class {cls}: {e}")
            continue

    # Calibration
    prob_pos = probs.max(axis=1)
    frac_pos, mean_pred = calibration_curve((y_test == y_pred).astype(int), prob_pos, n_bins=10, strategy="uniform")
    brier = brier_score_loss(y_test == y_pred, prob_pos)
    conf_bins = _confidence_bins(probs, y_test, n_bins=10)

    acc_ci = f1_ci = None
    if bootstrap:
        acc_ci = bootstrap_ci(lambda yt, yp: accuracy_score(yt, yp), y_test, y_pred, n_bootstrap)
        f1_ci = bootstrap_ci(lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0), y_test, y_pred, n_bootstrap)

    return {
        "overall": {
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1,
            "accuracy_ci": acc_ci,
            "f1_ci": f1_ci,
        },
        "per_class": report,
        "confusion_matrix": cm,
        "class_names": class_names,
        "roc": roc_data,
        "pr": pr_data,
        "calibration": {"fraction_of_positives": frac_pos, "mean_predicted": mean_pred, "brier": brier},
        "confidence_bins": conf_bins,
        "y_true": y_test,
        "y_pred": y_pred,
        "probs": probs,
    }


def export_results_json(results: Dict, path: Path) -> Path:
    payload = {
        "overall": results.get("overall"),
        "class_names": results.get("class_names"),
        "y_true": results.get("y_true", []).tolist(),
        "y_pred": results.get("y_pred", []).tolist(),
        "probs": results.get("probs", []).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def generate_report_html(results: Dict, dest: Path) -> Path:
    """Simple HTML report with metrics and confusion matrix."""
    cm = results["confusion_matrix"]
    classes = results["class_names"]
    overall = results["overall"]

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    img_path = dest.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(img_path, dpi=200)
    plt.close(fig)

    html = f"""
    <html><body>
    <h2>ThyroidAI Evaluation Report</h2>
    <p>Accuracy: {overall['accuracy']:.3f}, Macro F1: {overall['f1_macro']:.3f}</p>
    <img src="{img_path.name}" alt="Confusion Matrix"/>
    </body></html>
    """
    dest.write_text(html)
    return dest

