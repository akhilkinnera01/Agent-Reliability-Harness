"""
Score one or more judge models on the groundedness benchmark and persist the
per-case results, so plots can be regenerated for free afterwards.

    OPENAI_API_KEY=... python -m benchmarks.run_comparison gpt-4o-mini gpt-3.5-turbo

Writes benchmarks/results/<safe-model-name>.json per model:
    {model, auc, precision, recall, f1, correlation, n, scores[], labels[]}
"""

import json
import pathlib
import sys

from benchmarks import validate_groundedness
from benchmarks._judge import get_judge

RESULTS = pathlib.Path(__file__).parent / "results"


def _safe(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def run_one(model: str) -> dict:
    judge = get_judge(model)
    if judge is None:
        raise SystemExit(f"no API key for {model}")
    res = validate_groundedness.run(judge)
    res["model"] = model
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{_safe(model)}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"{model:34} AUC={res['auc']:.3f}  F1={res['f1']:.3f}  "
          f"P={res['precision']:.3f}  R={res['recall']:.3f}  (n={res['n']})")
    return res


if __name__ == "__main__":
    models = sys.argv[1:]
    if not models:
        raise SystemExit("usage: python -m benchmarks.run_comparison <model> [model ...]")
    for m in models:
        run_one(m)
