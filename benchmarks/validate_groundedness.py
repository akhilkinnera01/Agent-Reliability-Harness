"""
Validate groundedness against labeled grounded/hallucinated answers.

Scores each (source, answer) pair through the real pipeline — extract atomic
claims, then NLI-classify each against the source — and reports how well the
"fraction of claims entailed" separates grounded answers from hallucinations.

Live numbers need an API key (gemini/gemini-2.5-flash). Offline, __main__ uses a
deterministic stub judge to validate the plumbing and the grounded > hallucinated
ordering (not model quality).
"""

import json
import pathlib

from arh.core.agent_wrapper import AgentWrapper
from arh.core.models import AgentResponse
from arh.tests.groundedness import GroundednessTest
from benchmarks import metrics

DATA = pathlib.Path(__file__).parent / "datasets" / "groundedness_cases.json"


def score_case(gt: GroundednessTest, judge, source: str, answer: str) -> float:
    """Fraction of the answer's claims that are entailed by the source."""
    claims = gt._extract_claims(judge, answer)
    if not claims:
        return 1.0  # no factual claims -> nothing to hallucinate
    entailed = sum(
        1 for c in claims if gt._classify_claim(judge, c, source) == "ENTAILED"
    )
    return entailed / len(claims)


def run(judge) -> dict:
    cases = json.loads(DATA.read_text())["cases"]
    gt = GroundednessTest(judge=judge)
    scores = [score_case(gt, judge, c["source"], c["answer"]) for c in cases]
    labels = [1 if c["grounded"] else 0 for c in cases]
    thr, _ = metrics.best_threshold(scores, labels)
    p, r, f = metrics.prf(scores, labels, thr)
    return {
        "metric": "groundedness",
        "n": len(cases),
        "auc": round(metrics.roc_auc(scores, labels), 3),
        "best_threshold": round(thr, 3),
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(f, 3),
        "correlation": round(metrics.point_biserial(scores, labels), 3),
    }


class _StubJudge(AgentWrapper):
    """Deterministic offline judge: splits the answer into sentence-claims and
    entails a claim when most of its content words appear in the source. Crude,
    but enough to exercise the pipeline and the grounded>hallucinated ordering."""

    def __init__(self):
        super().__init__(endpoint="stub://", model="stub")

    def query(self, prompt: str, **kw) -> AgentResponse:
        if "atomic factual claims" in prompt:
            answer = prompt.split("TEXT:\n", 1)[-1]
            claims = [s.strip() for s in answer.replace("\n", " ").split(".") if s.strip()]
            return AgentResponse(content="\n".join(claims), latency_ms=0, model="stub")
        if "strict fact-checker" in prompt:
            source = prompt.split("SOURCE:\n", 1)[-1].split("\n\nCLAIM:")[0].lower()
            claim = prompt.split("CLAIM:\n", 1)[-1].lower()
            stop = {"the", "is", "a", "of", "and", "at", "in", "to", "as", "long",
                    "you", "can", "it", "for", "within", "with"}
            words = [w.strip(".,") for w in claim.split() if w.strip(".,") not in stop]
            words = [w for w in words if len(w) > 2]
            if not words:
                return AgentResponse(content="NOT_SUPPORTED", latency_ms=0, model="stub")
            hits = sum(1 for w in words if w in source)
            verdict = "ENTAILED" if hits / len(words) >= 0.7 else "NOT_SUPPORTED"
            return AgentResponse(content=verdict, latency_ms=0, model="stub")
        return AgentResponse(content="", latency_ms=0, model="stub")


if __name__ == "__main__":
    from benchmarks._judge import get_judge

    judge = get_judge()
    if judge is None:
        print("[live] skipped — set GEMINI_API_KEY for real groundedness numbers")
        print("[stub] validating pipeline plumbing + ordering...")
        judge = _StubJudge()
        cases = json.loads(DATA.read_text())["cases"]
        gt = GroundednessTest(judge=judge)
        grounded = [score_case(gt, judge, c["source"], c["answer"])
                    for c in cases if c["grounded"]]
        hallucinated = [score_case(gt, judge, c["source"], c["answer"])
                        for c in cases if not c["grounded"]]
        assert sum(grounded) / len(grounded) > sum(hallucinated) / len(hallucinated), \
            (grounded, hallucinated)
        print("OK")

    res = run(judge)
    for k, v in res.items():
        print(f"{k:>14}: {v}")
