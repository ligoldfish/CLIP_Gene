# scripts/metrics.py
from __future__ import annotations

import numpy as np


def average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    AP = sum (R_n - R_{n-1}) * P_n after sorting by score desc.
    If no positives, return 0.0
    """
    y_true = y_true.astype(np.float32)
    y_score = y_score.astype(np.float32)

    pos = float(y_true.sum())
    if pos <= 0:
        return 0.0

    order = np.argsort(-y_score)
    y = y_true[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)

    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(pos, 1e-12)

    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += float(p) * float(r - prev_r)
        prev_r = float(r)
    return float(ap)


def mean_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    y_true: [N,C] in {0,1}
    y_score:[N,C] in R
    """
    assert y_true.ndim == 2 and y_score.ndim == 2
    C = y_true.shape[1]
    aps = []
    for c in range(C):
        aps.append(average_precision_binary(y_true[:, c], y_score[:, c]))
    return float(np.mean(aps)) if len(aps) else 0.0
