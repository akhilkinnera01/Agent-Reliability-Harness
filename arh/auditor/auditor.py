"""
ARH Adversarial Auditor Module

Main orchestrator for adversarial documentation auditing.
Implements the Dr. Zero proposer-solver loop for finding doc flaws.
"""

from typing import List, Dict, Optional
from ..core.agent_wrapper import AgentWrapper
from ..core.models import AuditReport, Finding, FlawType, Severity
from .proposer import Proposer, HopComplexity
from .solver import Solver
from .evaluator import Evaluator
from datetime import datetime
import json


class AdversarialAuditor:
    """
    Main orchestrator for adversarial documentation auditing.
    Implements the Dr. Zero proposer-solver loop for finding doc flaws.
    """
    
    def __init__(
        self,
        proposer_model: AgentWrapper,
        solver_model: AgentWrapper = None,
        hop_complexity: List[HopComplexity] = None,
        flaw_types: List[FlawType] = None,
        answerability_gate: bool = True,
        severity_weights: Dict[Severity, float] = None,
        score_floor: float = 0.2
    ):
        """
        Initialize the adversarial auditor.

        Args:
            proposer_model: LLM wrapper for generating questions
            solver_model: LLM wrapper for answering (defaults to proposer_model)
            hop_complexity: Complexity levels to test
            flaw_types: Filter for specific flaw types (None = all)
            answerability_gate: If True, a NOT_FOUND/PARTIAL is only counted as a
                flaw when a complete doc should answer the question (soundness
                gate — suppresses out-of-scope false positives). Costs one extra
                model call per gap; disable for cheaper/looser audits.
            severity_weights: Per-severity score penalties. Defaults below.
            score_floor: Minimum overall score (penalty cap).
        """
        self.proposer = Proposer(proposer_model)
        self.solver = Solver(solver_model or proposer_model)
        self.evaluator = Evaluator()
        self.hop_complexity = hop_complexity or [
            HopComplexity.ONE,
            HopComplexity.TWO
        ]
        self.flaw_types = flaw_types  # Filter for specific flaws
        self.proposer_model = proposer_model
        self.answerability_gate = answerability_gate
        # ponytail: arbitrary-but-sane defaults. Calibrate against the Phase 3
        # labeled-docs benchmark; do not treat these as validated yet.
        self.severity_weights = severity_weights or {
            Severity.CRITICAL: 0.25,
            Severity.HIGH: 0.15,
            Severity.MEDIUM: 0.08,
            Severity.LOW: 0.03
        }
        self.score_floor = score_floor
    
    def audit(
        self,
        document: str,
        sections: List[Dict[str, str]] = None,
        document_name: str = "document"
    ) -> AuditReport:
        """
        Audit a document for flaws.
        
        Args:
            document: Full document text
            sections: Optional list of {"name": str, "content": str}
                     If not provided, treats entire doc as one section
            document_name: Name of the document for reporting
            
        Returns:
            AuditReport with findings and score
        """
        if sections is None:
            sections = [{"name": "full_document", "content": document}]
        
        all_findings: List[Finding] = []
        
        for section in sections:
            section_findings = self._audit_section(
                document=document,
                section_name=section["name"],
                section_content=section["content"]
            )
            all_findings.extend(section_findings)
        
        # Collapse the same gap surfaced by multiple questions into one finding.
        all_findings = self._dedup_findings(all_findings)

        # Filter by flaw type if specified
        if self.flaw_types:
            all_findings = [
                f for f in all_findings
                if f.flaw_type in self.flaw_types
            ]
        
        # Calculate overall score
        score = self._calculate_score(all_findings, document)
        
        return AuditReport(
            document=document_name,
            section="all",
            overall_score=score,
            findings=all_findings,
            timestamp=datetime.now()
        )
    
    def _audit_section(
        self,
        document: str,
        section_name: str,
        section_content: str
    ) -> List[Finding]:
        """Audit a single section of the document."""
        findings = []
        
        # Generate adversarial questions
        questions = self.proposer.generate_questions(
            document=document,
            section=section_content,
            hop_complexity=self.hop_complexity,
            questions_per_hop=3
        )
        
        # Test each question with solver
        for question in questions:
            q_text = question.get("question", "")
            solver_response = self.solver.answer(
                question=q_text,
                document=document
            )

            # Soundness gate: a gap-by-absence only counts if a complete doc
            # SHOULD answer it. Skips out-of-scope false positives.
            if (self.answerability_gate
                    and solver_response.status in ("NOT_FOUND", "PARTIAL")
                    and not self.proposer.is_answerable(q_text, document)):
                continue

            # Evaluate for flaw
            finding = self.evaluator.evaluate(
                question=question,
                solver_response=solver_response,
                section_text=section_content
            )

            if finding:
                findings.append(finding)

        return findings

    @staticmethod
    def _dedup_findings(findings: List[Finding]) -> List[Finding]:
        """Collapse findings that point at the same flaw at the same line."""
        seen = {}
        for f in findings:
            seen.setdefault((f.flaw_type, f.line), f)
        return list(seen.values())
    
    def _calculate_score(self, findings: List[Finding], document: str) -> float:
        """Calculate document reliability score based on findings."""
        if not findings:
            return 1.0

        total_penalty = sum(
            self.severity_weights.get(f.severity, 0.05)
            for f in findings
        )

        score = max(self.score_floor, 1.0 - total_penalty)

        return round(score, 3)
    
    def audit_file(self, filepath: str) -> AuditReport:
        """
        Convenience method to audit a file directly.
        
        Args:
            filepath: Path to the document file
            
        Returns:
            AuditReport for the file
        """
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Use filename as document name
        import os
        doc_name = os.path.basename(filepath)
        
        return self.audit(content, document_name=doc_name)
    
    def audit_simple(self, document: str, document_name: str = "document") -> AuditReport:
        """
        DEMO-ONLY — NOT A RELIABILITY SIGNAL. Keyword-matching audit with no
        LLM; flaw findings are illustrative only. Use audit() with an LLM
        proposer for real results.

        Args:
            document: Document text to audit
            document_name: Name for reporting
            
        Returns:
            AuditReport based on simple keyword analysis
        """
        findings = []
        
        # Generate simple questions
        questions = self.proposer.generate_questions_simple(document[:500], num_questions=5)
        
        # Answer using simple matching
        for question in questions:
            solver_response = self.solver.answer_simple(
                question=question.get("question", ""),
                document=document
            )
            
            # Evaluate for flaw
            finding = self.evaluator.evaluate(
                question=question,
                solver_response=solver_response,
                section_text=document[:500]
            )
            
            if finding:
                findings.append(finding)
        
        score = self._calculate_score(findings, document)
        
        return AuditReport(
            document=document_name,
            section="all",
            overall_score=score,
            findings=findings,
            timestamp=datetime.now()
        )
    
    def generate_report_dict(self, report: AuditReport) -> Dict:
        """
        Convert AuditReport to a dictionary for JSON serialization.
        
        Args:
            report: AuditReport to convert
            
        Returns:
            Dictionary representation
        """
        return {
            "document": report.document,
            "section": report.section,
            "overall_score": report.overall_score,
            "timestamp": report.timestamp.isoformat(),
            "findings_count": len(report.findings),
            "findings": [
                {
                    "line": f.line,
                    "text": f.text,
                    "flaw_type": f.flaw_type.value,
                    "severity": f.severity.value,
                    "question": f.question,
                    "recommendation": f.recommendation
                }
                for f in report.findings
            ],
            "summary": self.evaluator.summarize_findings(report.findings)
        }
    
    def print_report(self, report: AuditReport):
        """Print a human-readable report to console."""
        print("\n" + "=" * 60)
        print("DOCUMENT AUDIT REPORT")
        print("=" * 60)
        print(f"Document: {report.document}")
        print(f"Score: {report.overall_score:.1%}")
        print(f"Findings: {len(report.findings)}")
        print("-" * 60)
        
        if not report.findings:
            print("✅ No documentation flaws detected!")
        else:
            for i, finding in enumerate(report.findings, 1):
                severity_icons = {
                    Severity.CRITICAL: "🚨",
                    Severity.HIGH: "⚠️",
                    Severity.MEDIUM: "📝",
                    Severity.LOW: "💡"
                }
                icon = severity_icons.get(finding.severity, "•")
                print(f"\n{icon} Finding {i}: {finding.flaw_type.value}")
                print(f"   Line {finding.line}: {finding.text[:60]}...")
                print(f"   Question: {finding.question[:60]}...")
                print(f"   Recommendation: {finding.recommendation}")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    # ponytail: end-to-end check on a scripted stub — fails if the soundness
    # gate or dedup regress. No real LLM/API needed.
    from ..core.models import AgentResponse, Finding, FlawType

    class _Stub(AgentWrapper):
        def __init__(self):
            super().__init__(endpoint="stub://", model="stub")
        def query(self, prompt: str, **kw) -> AgentResponse:
            if "adversarial documentation auditor" in prompt:
                return AgentResponse(content=(
                    "Q1: What safety precautions apply?\nTARGET: safety\n"
                    "FLAW_IF_MISSING: SAFETY_GAP\n"
                    "Q2: What is the founder's favorite color?\nTARGET: trivia\n"
                    "FLAW_IF_MISSING: AMBIGUOUS"
                ), latency_ms=0, model="stub")
            if "SHOULD a complete" in prompt:  # answerability gate
                verdict = "YES" if "precaution" in prompt.lower() else "NO"
                return AgentResponse(content=verdict, latency_ms=0, model="stub")
            if "STRICT documentation validator" in prompt:  # solver
                return AgentResponse(content="STATUS: NOT_FOUND\nCONFIDENCE: 10",
                                     latency_ms=0, model="stub")
            return AgentResponse(content="", latency_ms=0, model="stub")

    # Pure dedup check.
    g = lambda: Finding(line=3, text="t", flaw_type=FlawType.SAFETY_GAP,
                        severity=Severity.CRITICAL, question="q",
                        solver_response="", recommendation="")
    assert len(AdversarialAuditor._dedup_findings([g(), g()])) == 1

    doc = "Lab procedure: mix reagent A with B. Heat the mixture to 50 degrees."
    report = AdversarialAuditor(proposer_model=_Stub()).audit(doc, document_name="lab.md")
    # Out-of-scope trivia suppressed by the gate; the safety gap survives, and the
    # two hops collapse to one finding via dedup.
    assert len(report.findings) == 1, [(x.flaw_type, x.line) for x in report.findings]
    assert report.findings[0].flaw_type == FlawType.SAFETY_GAP, report.findings[0]
    print("OK")
