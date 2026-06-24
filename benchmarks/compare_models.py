"""
Run the labeled benchmarks across several judge models and print them side by
side. The point is to show the harness *discriminates*: a reliability layer is
only worth anything if a weaker model scores measurably worse than a stronger one
on the same data.

    OPENAI_API_KEY=... python -m benchmarks.compare_models gpt-4o-mini gpt-3.5-turbo

Defaults to gpt-4o-mini vs gpt-3.5-turbo. Numbers are whatever the run produces;
nothing here is tuned to a target.
"""

import sys

from benchmarks import validate_groundedness
from benchmarks._judge import get_judge
from benchmarks.check_auditor_acceptance import evaluate as auditor_eval


def run_one(model: str) -> dict:
    judge = get_judge(model)
    gnd = validate_groundedness.run(judge)

    from arh.auditor.auditor import AdversarialAuditor
    from arh.auditor.proposer import HopComplexity
    card = auditor_eval(AdversarialAuditor(
        proposer_model=judge, hop_complexity=[HopComplexity.ONE]))

    return {
        "model": model,
        "gnd_auc": gnd["auc"],
        "gnd_p": gnd["precision"],
        "gnd_r": gnd["recall"],
        "gnd_f1": gnd["f1"],
        "gnd_corr": gnd["correlation"],
        "aud_recall": round(card.recall, 3),
        "aud_fp": card.clean_findings,
    }


def main() -> int:
    models = sys.argv[1:] or ["gpt-4o-mini", "gpt-3.5-turbo"]
    rows = [run_one(m) for m in models]

    cols = [("model", 16), ("gnd_auc", 9), ("gnd_p", 7), ("gnd_r", 7),
            ("gnd_f1", 7), ("gnd_corr", 9), ("aud_recall", 11), ("aud_fp", 7)]
    print("\nMODEL COMPARISON  (groundedness n=%d cases, auditor seeded-flaw doc)"
          % validate_groundedness_n())
    print("=" * sum(w for _, w in cols))
    print("".join(f"{name:<{w}}" for name, w in cols))
    print("-" * sum(w for _, w in cols))
    for r in rows:
        print("".join(f"{str(r[name]):<{w}}" for name, w in cols))
    print("=" * sum(w for _, w in cols))

    if len(rows) == 2:
        d = round(rows[0]["gnd_auc"] - rows[1]["gnd_auc"], 3)
        print(f"groundedness AUC delta ({rows[0]['model']} - {rows[1]['model']}): {d:+}")
    return 0


def validate_groundedness_n() -> int:
    import json, pathlib
    p = pathlib.Path(validate_groundedness.DATA)
    return len(json.loads(p.read_text())["cases"])


if __name__ == "__main__":
    sys.exit(main())
