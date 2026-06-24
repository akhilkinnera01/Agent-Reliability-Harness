"""
Meta-evaluation metrics — how well does an ARH score track the ground-truth
label? Pure numpy (no sklearn): ROC-AUC (tie-aware), precision/recall/F1 at a
threshold, best-F1 threshold search, and point-biserial correlation.
"""

import numpy as np


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share the mean rank — like scipy.rankdata."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def roc_auc(scores, labels) -> float:
    """Rank-based ROC-AUC (== P(score_pos > score_neg)). NaN if one class empty."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def prf(scores, labels, threshold: float):
    """Precision, recall, F1 for `score >= threshold` as the positive prediction."""
    pred = np.asarray(scores, dtype=float) >= threshold
    labels = np.asarray(labels, dtype=bool)
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def best_threshold(scores, labels):
    """Threshold (a midpoint between observed scores) maximizing F1."""
    uniq = sorted(set(float(s) for s in scores))
    # Candidate cut points: just below the lowest, and midpoints between scores.
    cands = [uniq[0] - 1e-6] + [(uniq[i] + uniq[i + 1]) / 2 for i in range(len(uniq) - 1)]
    best_t, best_f = cands[0], -1.0
    for t in cands:
        _, _, f = prf(scores, labels, t)
        if f > best_f:
            best_t, best_f = t, f
    return best_t, best_f


def point_biserial(scores, labels) -> float:
    """Correlation of score with the binary label (Pearson). NaN if degenerate."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if scores.std() == 0 or labels.std() == 0:
        return float("nan")
    return float(np.corrcoef(scores, labels)[0, 1])


if __name__ == "__main__":
    # ponytail: smallest check that fails if the meta-metrics regress.
    s = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
    y = [1, 1, 1, 0, 0, 0]
    assert roc_auc(s, y) == 1.0
    t, f = best_threshold(s, y)
    assert f == 1.0 and 0.2 < t < 0.7, (t, f)
    assert roc_auc([1, 1, 0, 0], [1, 0, 1, 0]) == 0.5  # random -> 0.5
    assert abs(point_biserial(s, y)) > 0.9
    print("OK")
