"""
ARH meta-evaluation runner — the part that earns "production".

Runs ARH's own metrics against labeled data and reports the numbers ARH itself
should be judged on: AUC / precision / recall of "ARH flags it" vs "it is
actually bad". Prints one scorecard and exits non-zero if an *enforceable*
metric regresses below its recorded baseline (CI gate).

A metric is only enforced when its proper backend is available:
  - similarity   -> requires sentence-transformers (else informational only)
  - groundedness -> requires an API key (GEMINI_API_KEY)
  - auditor      -> requires an API key

Run:
    python -m benchmarks.validate                 # similarity only, offline
    GEMINI_API_KEY=... python -m benchmarks.validate   # full suite
"""

import sys

from benchmarks import validate_similarity, validate_groundedness
from benchmarks._judge import get_judge, have_key
from benchmarks.check_auditor_acceptance import evaluate as auditor_eval

# Recorded baselines = achieved AUC minus a small margin, used as a regression
# floor (not an aspirational target). Re-baseline via benchmarks/calibrate.py
# when the datasets or backend change. Achieved on all-MiniLM-L6-v2:
# similarity AUC 0.72 (contradiction pairs are the residual errors).
BASELINES = {"similarity": 0.70, "groundedness": 0.70}


def _embeddings_available() -> bool:
    from arh.core.similarity import _get_model
    return _get_model() is not None


def main() -> int:
    rows = []          # (name, auc-or-score, detail, enforced, ok)
    failures = []

    # --- similarity (robustness/consistency backbone) ---
    sim = validate_similarity.run()
    enforced = _embeddings_available()
    ok = (not enforced) or sim["auc"] >= BASELINES["similarity"]
    rows.append(("similarity", sim["auc"],
                 f"P={sim['precision']} R={sim['recall']} thr={sim['best_threshold']}",
                 enforced, ok))
    if enforced and not ok:
        failures.append(f"similarity AUC {sim['auc']} < {BASELINES['similarity']}")

    # --- groundedness + auditor (need a live LLM) ---
    judge = get_judge()
    if judge is not None:
        gnd = validate_groundedness.run(judge)
        ok = gnd["auc"] >= BASELINES["groundedness"]
        rows.append(("groundedness", gnd["auc"],
                     f"P={gnd['precision']} R={gnd['recall']} thr={gnd['best_threshold']}",
                     True, ok))
        if not ok:
            failures.append(f"groundedness AUC {gnd['auc']} < {BASELINES['groundedness']}")

        from arh.auditor.auditor import AdversarialAuditor
        from arh.auditor.proposer import HopComplexity
        card = auditor_eval(AdversarialAuditor(
            proposer_model=judge, hop_complexity=[HopComplexity.ONE]))
        rows.append(("auditor", round(card.recall, 3),
                     f"recall={card.recall:.2f} clean_FP={card.clean_findings}",
                     True, card.passed))
        if not card.passed:
            failures.append(f"auditor acceptance failed: {card}")
    else:
        rows.append(("groundedness", "—", "skipped (no API key)", False, True))
        rows.append(("auditor", "—", "skipped (no API key)", False, True))

    # --- scorecard ---
    print("\nARH VALIDATION SCORECARD")
    print("=" * 64)
    print(f"{'metric':<14}{'auc/score':<12}{'detail':<30}{'gate'}")
    print("-" * 64)
    for name, val, detail, enforced, ok in rows:
        gate = "—" if not enforced else ("PASS" if ok else "FAIL")
        print(f"{name:<14}{str(val):<12}{detail:<30}{gate}")
    print("=" * 64)
    if not _embeddings_available():
        print("note: similarity on Jaccard fallback (informational). "
              "Install arh[semantic] for the enforced number.")
    if not have_key():
        print("note: groundedness/auditor skipped — set GEMINI_API_KEY for live numbers.")

    if failures:
        print("\nREGRESSIONS:")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
