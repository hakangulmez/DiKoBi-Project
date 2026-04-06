"""
Evaluation metrics for classification tasks.
Includes accuracy, MSE, MAE, and Quadratic Weighted Kappa (QWK).
"""

import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    cohen_kappa_score,
    confusion_matrix,
)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Quadratic Weighted Kappa (QWK).
    
    QWK is commonly used for ordinal classification tasks where
    predictions close to the true value are penalized less than
    predictions far from the true value.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        QWK score (0 = random, 1 = perfect agreement)
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary with all metrics
    """
    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }
    
    # Add confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm
    
    # Calculate per-class accuracy
    n_classes = len(np.unique(y_true))
    for i in range(n_classes):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
            metrics[f"accuracy_class_{i}"] = class_acc
    
    return metrics


def print_metrics_summary(metrics: Dict[str, any], title: Optional[str] = None):
    """
    Print formatted metrics summary.
    
    Args:
        metrics: Dictionary from calculate_metrics()
        title: Optional title to print
    """
    if title:
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
    
    # Main metrics
    print(f"\nAccuracy:  {metrics['accuracy']:.2%}")
    print(f"QWK:       {metrics['qwk']:.4f}")
    print(f"MSE:       {metrics['mse']:.4f}")
    print(f"MAE:       {metrics['mae']:.4f}")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for key in sorted(metrics.keys()):
        if key.startswith("accuracy_class_"):
            class_id = key.split("_")[-1]
            print(f"  Class {class_id}: {metrics[key]:.2%}")
    
    # Confusion matrix
    if "confusion_matrix" in metrics:
        print("\nConfusion Matrix:")
        cm = metrics["confusion_matrix"]
        print(cm)
    
    print("=" * 60 + "\n")


def compare_metrics(
    metrics_list: list,
    labels: list,
    sort_by: str = "qwk"
) -> None:
    """
    Compare multiple sets of metrics side-by-side.
    
    Args:
        metrics_list: List of metric dictionaries
        labels: Labels for each metric set
        sort_by: Metric to sort by
    """
    # Create sorted indices
    sort_values = [m[sort_by] for m in metrics_list]
    sorted_indices = np.argsort(sort_values)[::-1]  # Descending
    
    print("\n" + "=" * 80)
    print("METRICS COMPARISON")
    print("=" * 80)
    print(f"\n{'Rank':<6} {'Name':<25} {'Accuracy':<12} {'QWK':<12} {'MSE':<12} {'MAE':<12}")
    print("-" * 80)
    
    for rank, idx in enumerate(sorted_indices, 1):
        label = labels[idx]
        m = metrics_list[idx]
        print(f"{rank:<6} {label:<25} {m['accuracy']:<12.2%} {m['qwk']:<12.4f} "
              f"{m['mse']:<12.4f} {m['mae']:<12.4f}")
    
    print("=" * 80 + "\n")
