"""
Phase 2 acceptance check for the Adversarial Auditor.

Criteria (from the implementation plan):
  - On a deliberately flawed doc, the auditor finds the seeded flaws
    (recall) with a low false-positive rate.
  - On a known-good doc, it produces near-zero findings.

Live numbers require an API key (GEMINI_API_KEY -> gemini/gemini-2.5-flash).
Run live with:

    GEMINI_API_KEY=... python -m benchmarks.check_auditor_acceptance

Without a key, __main__ runs a deterministic stub that validates the
measurement logic (so CI stays offline and green).
"""

import json
import pathlib
from dataclasses import dataclass
from typing import List

from arh.core.models import AuditReport, Finding, FlawType, Severity

HERE = pathlib.Path(__file__).parent / "auditor"

# Acceptance thresholds.
MIN_RECALL = 0.66        # find at least 2 of 3 seeded flaw types
MAX_CLEAN_FINDINGS = 1   # "near-zero" on the clean doc


@dataclass
class Scorecard:
    recall: float
    seeded_types: set
    found_types: set
    clean_findings: int

    @property
    def passed(self) -> bool:
        return self.recall >= MIN_RECALL and self.clean_findings <= MAX_CLEAN_FINDINGS

    def __str__(self) -> str:
        return (
            f"recall={self.recall:.2f} (found {sorted(self.found_types & self.seeded_types)} "
            f"of {sorted(self.seeded_types)})\n"
            f"clean-doc false positives={self.clean_findings} "
            f"(max allowed {MAX_CLEAN_FINDINGS})\n"
            f"ACCEPTANCE: {'PASS' if self.passed else 'FAIL'}"
        )


def evaluate(auditor) -> Scorecard:
    """Run `auditor` against the labeled docs and score it. `auditor` just
    needs an `.audit(text, document_name=...)` method returning an AuditReport."""
    gold = json.loads((HERE / "seeded_flaws.json").read_text())
    seeded_types = {s["flaw_type"] for s in gold["seeded"]}

    flawed = (HERE / "flawed_manual.md").read_text()
    clean = (HERE / "clean_manual.md").read_text()

    flawed_report = auditor.audit(flawed, document_name="flawed_manual.md")
    found_types = {f.flaw_type.value for f in flawed_report.findings}
    recall = len(seeded_types & found_types) / len(seeded_types)

    clean_report = auditor.audit(clean, document_name="clean_manual.md")

    return Scorecard(
        recall=recall,
        seeded_types=seeded_types,
        found_types=found_types,
        clean_findings=len(clean_report.findings),
    )


def _finding(flaw: FlawType) -> Finding:
    return Finding(line=1, text="", flaw_type=flaw, severity=Severity.HIGH,
                   question="", solver_response="", recommendation="")


class _StubAuditor:
    """Deterministic stand-in: returns the seeded flaws for the flawed doc and
    nothing for the clean doc. Validates the scoring logic without an LLM."""

    def audit(self, document: str, document_name: str = "") -> AuditReport:
        if "flawed" in document_name:
            findings = [_finding(FlawType.SAFETY_GAP),
                        _finding(FlawType.MISSING_PREREQ),
                        _finding(FlawType.AMBIGUOUS)]
        else:
            findings = []
        return AuditReport(document=document_name, section="all",
                           overall_score=0.0, findings=findings)


if __name__ == "__main__":
    # Offline logic check (always runs): the harness scores a perfect stub.
    card = evaluate(_StubAuditor())
    assert card.passed, card
    assert card.recall == 1.0 and card.clean_findings == 0, card
    print("[stub] measurement logic OK")
    print(card)

    # Live check (only with an API key): score the real LLM-backed auditor.
    from benchmarks._judge import get_judge
    judge = get_judge()
    if judge is not None:
        from arh.auditor.auditor import AdversarialAuditor
        print("\n[live] running gemini-backed auditor...")
        live = evaluate(AdversarialAuditor(proposer_model=judge))
        print(live)
        assert live.passed, f"Phase 2 acceptance FAILED: {live}"
    else:
        print("\n[live] skipped — set GEMINI_API_KEY for real acceptance numbers")
