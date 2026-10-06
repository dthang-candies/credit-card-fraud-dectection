"""
Evaluation Metrics Module for Credit Card Fraud Detection.
Implements comprehensive evaluation functions matching the metrics reported in the paper:
- Accuracy
- AROC Score (ROC-AUC)
- Precision
- Recall
- F1-Score
- Confusion Matrix (TP, FP, FN, TN)
"""

from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    y_prob: Optional[Union[np.ndarray, pd.Series, list]] = None,
    pos_label: int = 1,
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics for fraud detection.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted probability or anomaly score for positive class.
        pos_label: Positive class label (default 1 for fraud).

    Returns:
        dict with Accuracy, AROC, Precision, Recall, F1, TP, FP, FN, TN.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    # Standard confusion matrix where 1 is fraud (positive) and 0 is legitimate (negative)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

    # ROC-AUC score
    aroc = None
    pr_auc = None
    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob).ravel()
        try:
            aroc = float(roc_auc_score(y_true, y_prob_arr))
            pr_auc = float(average_precision_score(y_true, y_prob_arr))
        except ValueError:
            # Fallback if only 1 class present in batch
            aroc = float(roc_auc_score(y_true, y_pred))
            pr_auc = float(average_precision_score(y_true, y_pred))
    else:
        try:
            aroc = float(roc_auc_score(y_true, y_pred))
            pr_auc = float(average_precision_score(y_true, y_pred))
        except ValueError:
            aroc = 0.5
            pr_auc = 0.0

    return {
        "Accuracy": round(float(acc), 4),
        "AROC Score": round(float(aroc), 4) if aroc is not None else 0.0,
        "PR-AUC": round(float(pr_auc), 4) if pr_auc is not None else 0.0,
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1- Score": round(float(f1), 4),
        "True Positive": int(tp),
        "False Positive": int(fp),
        "False Negative": int(fn),
        "True Negative": int(tn),
    }


def compute_paper_style_metrics(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    y_prob: Optional[Union[np.ndarray, pd.Series, list]] = None,
) -> Dict[str, Any]:
    """
    Format metrics matching Table 4 & Table 5 in the paper:
    (In the paper's confusion matrix convention:
     - Normal class is treated as positive in the table display:
       TP = correctly detected normal, TN = correctly detected fraud,
       FP = normal misclassified as fraud, FN = fraud misclassified as normal)
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob, pos_label=1)

    # Add paper display mapping
    metrics["Paper_TP (Normal Đúng)"] = int(tn)
    metrics["Paper_FP (Bình thường nhầm thành gian lận)"] = int(fp)
    metrics["Paper_FN (Gian lận bị bỏ sót)"] = int(fn)
    metrics["Paper_TN (Gian lận Đúng)"] = int(tp)

    return metrics


def print_metrics_report(metrics: Dict[str, Any], model_name: str = "Mô hình") -> None:
    """
    Print an aligned evaluation report to stdout.
    """
    print("-" * 55)
    print(f"KẾT QUẢ ĐÁNH GIÁ: {model_name.upper()}")
    print("-" * 55)
    print(f"Accuracy         : {metrics.get('Accuracy', 0):.4f}")
    print(f"AROC Score       : {metrics.get('AROC Score', 0):.4f}")
    print(f"PR-AUC           : {metrics.get('PR-AUC', 0):.4f}")
    print(f"Precision        : {metrics.get('Precision', 0):.4f}")
    print(f"Recall           : {metrics.get('Recall', 0):.4f}")
    print(f"F1- Score        : {metrics.get('F1- Score', 0):.4f}")
    print(f"True Positive (TP) : {metrics.get('True Positive', 0):,}")
    print(f"False Positive (FP): {metrics.get('False Positive', 0):,}")
    print(f"False Negative (FN): {metrics.get('False Negative', 0):,}")
    print(f"True Negative (TN) : {metrics.get('True Negative', 0):,}")
    print("-" * 55)


def compare_models(
    results: Dict[str, Dict[str, Any]],
    dataset_title: str = "TỔNG HỢP SO SÁNH CÁC MÔ HÌNH"
) -> pd.DataFrame:
    """
    Create a comparative DataFrame from multiple models' metric dictionaries.
    """
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        row.update(metrics)
        rows.append(row)

    df_comp = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print(f"{dataset_title}")
    print("=" * 80)
    columns_order = [
        "Model", "Accuracy", "AROC Score", "Precision", "Recall", "F1- Score",
        "True Positive", "False Positive", "False Negative", "True Negative"
    ]
    existing_cols = [c for c in columns_order if c in df_comp.columns]
    print(df_comp[existing_cols].to_string(index=False))
    print("=" * 80 + "\n")
    return df_comp
