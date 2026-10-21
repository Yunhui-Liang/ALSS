"""Internal and external clustering metrics."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def sse_for_indices(X: np.ndarray, indices: np.ndarray) -> float:
    """Within-cluster sum of squared Euclidean distances for selected rows."""
    if len(indices) == 0:
        return 0.0
    points = X[np.asarray(indices, dtype=np.int64)]
    center = np.mean(points, axis=0)
    residual = points - center
    return float(np.einsum("ij,ij->", residual, residual))


def cluster_sse(X: np.ndarray, labels: np.ndarray) -> float:
    """Total within-cluster SSE for a full label vector."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    if X.ndim != 2 or labels.ndim != 1 or X.shape[0] != labels.size:
        raise ValueError("X must be 2D and labels must contain one value per row")
    return float(sum(sse_for_indices(X, np.flatnonzero(labels == value)) for value in np.unique(labels)))


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Clustering accuracy after optimal one-to-one label matching."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.size == 0 or y_true.size != y_pred.size:
        raise ValueError("y_true and y_pred must have the same non-zero length")
    true_values, true_inverse = np.unique(y_true, return_inverse=True)
    pred_values, pred_inverse = np.unique(y_pred, return_inverse=True)
    contingency = np.zeros((true_values.size, pred_values.size), dtype=np.int64)
    np.add.at(contingency, (true_inverse, pred_inverse), 1)
    rows, columns = linear_sum_assignment(contingency, maximize=True)
    return float(contingency[rows, columns].sum() / y_true.size)


def evaluate_clustering(
    X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute SSE, ACC, NMI and ARI for one clustering."""
    return {
        "sse": cluster_sse(X, y_pred),
        "acc": clustering_accuracy(y_true, y_pred),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
    }
