"""
Calibrate ARH's metric thresholds against the labeled benchmarks and record
them, so every default cutoff is chosen by maximizing benchmark agreement rather
than by vibes.

Writes benchmarks/thresholds.json with the best-F1 threshold and the
precision/recall it achieves, per metric. Whatever backend is available is what
gets calibrated (and recorded), so the file never lies about how it was made.

    python -m benchmarks.calibrate                  # similarity only
    GEMINI_API_KEY=... python -m benchmarks.calibrate   # + groundedness
"""

import json
import pathlib

from benchmarks import validate_similarity, validate_groundedness
from benchmarks._judge import get_judge

OUT = pathlib.Path(__file__).parent / "thresholds.json"


def _backend() -> str:
    from arh.core.similarity import _get_model
    return "sentence-transformers (all-MiniLM-L6-v2)" if _get_model() else "jaccard-fallback"


def calibrate() -> dict:
    result = {}

    sim = validate_similarity.run()
    result["similarity"] = {
        "threshold": sim["best_threshold"],
        "precision": sim["precision"],
        "recall": sim["recall"],
        "auc": sim["auc"],
        "n": sim["n"],
        "backend": _backend(),
    }

    judge = get_judge()
    if judge is not None:
        gnd = validate_groundedness.run(judge)
        result["groundedness"] = {
            "threshold": gnd["best_threshold"],
            "precision": gnd["precision"],
            "recall": gnd["recall"],
            "auc": gnd["auc"],
            "n": gnd["n"],
            "backend": judge.model,
        }

    return result


if __name__ == "__main__":
    result = calibrate()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT}")
    print(json.dumps(result, indent=2))
