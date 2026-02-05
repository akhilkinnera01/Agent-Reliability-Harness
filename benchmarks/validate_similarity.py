"""
Validate the semantic-similarity backbone (robustness/consistency) against
labeled equivalent/non-equivalent pairs.

Reports AUC, the best-F1 threshold, and the precision/recall it achieves — the
numbers that justify (or refute) the 0.85 default cutoff.

Backend note: with sentence-transformers installed these are real semantic
numbers; on the Jaccard fallback they are degraded (and that gap is exactly the
argument for installing `arh[semantic]`).
"""

import json
import pathlib

from arh.core.similarity import semantic_similarity
from benchmarks import metrics

DATA = pathlib.Path(__file__).parent / "datasets" / "similarity_pairs.json"


def run() -> dict:
    pairs = json.loads(DATA.read_text())["pairs"]
    scores = [semantic_similarity(p["a"], p["b"]) for p in pairs]
    labels = [1 if p["equivalent"] else 0 for p in pairs]
    thr, _ = metrics.best_threshold(scores, labels)
    p, r, f = metrics.prf(scores, labels, thr)
    return {
        "metric": "similarity",
        "n": len(pairs),
        "auc": round(metrics.roc_auc(scores, labels), 3),
        "best_threshold": round(thr, 3),
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f, 3),
        "correlation": round(metrics.point_biserial(scores, labels), 3),
    }


if __name__ == "__main__":
    res = run()
    for k, v in res.items():
        print(f"{k:>14}: {v}")
