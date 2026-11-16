
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from src.config import LABELS


def compute_metrics(y_true, y_pred, labels=LABELS) -> dict:
    """All required metrics as a JSON-serialisable dict.

    ``confusion_matrix`` rows are TRUE labels and columns are PREDICTED labels,
    both in the order given by ``labels``.
    """
    p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_mac),
        "recall_macro": float(r_mac),
        "f1_macro": float(f_mac),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f_w),
        "labels": list(labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "classification_report_text": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, digits=4),
    }
