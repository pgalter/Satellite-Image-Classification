"""Evaluation metrics shared by every experiment."""
from __future__ import annotations

import numpy as np

from . import config as C


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int = C.N_CLASSES) -> np.ndarray:
    """Counts; rows = true class, columns = predicted class."""
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Overall accuracy (the paper's metric) plus class-level precision / recall / F1."""
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1)
    predicted = cm.sum(0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "recall": recall.tolist(),
        "precision": precision.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }
