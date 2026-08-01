"""Evaluation metrics, implemented in pure torch (no numpy/sklearn available)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def roc_auc(positive: Tensor, negative: Tensor) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    AUC = P(score(positive) > score(negative)), with ties counted as 0.5.
    Rank-based rather than threshold-sweeping, so it is exact and cheap.
    """
    pos = positive.reshape(-1).double()
    neg = negative.reshape(-1).double()
    n_pos, n_neg = pos.numel(), neg.numel()
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    combined = torch.cat([pos, neg])
    order = combined.argsort()
    ranks = torch.empty_like(combined)
    ranks[order] = torch.arange(1, combined.numel() + 1, dtype=torch.double)

    # Average ranks within tied groups so ties contribute 0.5.
    unique, inverse, counts = combined.unique(
        return_inverse=True, return_counts=True
    )
    if int(counts.max()) > 1:
        summed = torch.zeros_like(unique).scatter_add_(0, inverse, ranks)
        ranks = (summed / counts)[inverse]

    rank_sum = ranks[:n_pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@dataclass
class ClassificationReport:
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    def __str__(self) -> str:
        return (
            f"P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f}  "
            f"(tp={self.true_positive} fp={self.false_positive} "
            f"fn={self.false_negative} tn={self.true_negative})"
        )


def classification(predicted: list[bool], actual: list[bool]) -> ClassificationReport:
    tp = sum(p and a for p, a in zip(predicted, actual))
    fp = sum(p and not a for p, a in zip(predicted, actual))
    fn = sum((not p) and a for p, a in zip(predicted, actual))
    tn = sum((not p) and (not a) for p, a in zip(predicted, actual))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ClassificationReport(precision, recall, f1, tp, fp, fn, tn)
