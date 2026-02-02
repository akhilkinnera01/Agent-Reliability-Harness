"""
ARH Evaluator Module

Evaluates solver responses to classify documentation flaws.
Maps failures to actionable flaw types with severity levels.
"""

from typing import Dict, List, Optional
from ..core.models import FlawType, Severity, Finding
from .solver import SolverResponse


class Evaluator:
    """
    Evaluates solver responses to classify documentation flaws.
    """
    
    def __init__(self):
        """Initialize the evaluator with severity mappings."""
        self.severity_map = {
            FlawType.SAFETY_GAP: Severity.CRITICAL,
            FlawType.MISSING_PREREQ: Severity.HIGH,
            FlawType.CONTRADICTION: Severity.HIGH,
            FlawType.AMBIGUOUS: Severity.MEDIUM,
            FlawType.IMPLICIT_ASSUMPTION: Severity.MEDIUM,
            FlawType.TEMPORAL_GAP: Severity.LOW
        }
    
    def evaluate(
        self,
        question: Dict,
        solver_response: SolverResponse,
        section_text: str
    ) -> Optional[Finding]:
        """
        Evaluate if solver failure indicates a documentation flaw.
        
        Args:
            question: Question dictionary from proposer
            solver_response: Response from solver
            section_text: The section text being audited
            
        Returns:
            Finding if flaw detected, None otherwise
        """
        # If solver answered confidently, no flaw
        if solver_response.status == "FOUND" and solver_response.confidence > 0.8:
            return None
        
        # Determine flaw type
        flaw_type = self._classify_flaw(question, solver_response)
        
        if flaw_type is None:
            return None
        
        # Find relevant line in section
        line_num = self._find_relevant_line(section_text, question.get('question', ''))
        relevant_text = self._extract_relevant_text(section_text, line_num)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(flaw_type, question, solver_response)
        
        return Finding(
            line=line_num,
            text=relevant_text,
            flaw_type=flaw_type,
            severity=self.severity_map.get(flaw_type, Severity.MEDIUM),
            question=question.get('question', ''),
            solver_response=solver_response.raw_response[:200],
            recommendation=recommendation
        )
    
    # Proposer's FLAW_IF_MISSING string -> FlawType, by enum name.
    _EXPECTED_FLAW_MAP = {ft.name: ft for ft in FlawType}

    def _classify_flaw(self, question: Dict, response: SolverResponse) -> Optional[FlawType]:
        """Classify the type of documentation flaw."""

        # Use expected flaw if provided by proposer
        expected = question.get('expected_flaw', '').strip().upper()
        question_text = question.get('question', '').lower()

        if response.status == "NOT_FOUND":
            # Safety questions are critical regardless of the proposer's tag.
            safety_keywords = ['safety', 'hazard', 'danger', 'warning', 'caution', 'risk', 'harm']
            if any(word in question_text for word in safety_keywords):
                return FlawType.SAFETY_GAP
            # Honor the proposer's expected flaw across the full FlawType set,
            # defaulting to MISSING_PREREQ when it gave nothing usable.
            return self._EXPECTED_FLAW_MAP.get(expected, FlawType.MISSING_PREREQ)

        elif response.status == "AMBIGUOUS":
            return FlawType.AMBIGUOUS
        
        elif response.status == "PARTIAL":
            if response.missing_info:
                return FlawType.IMPLICIT_ASSUMPTION
            return FlawType.TEMPORAL_GAP
        
        elif response.confidence < 0.5:
            return FlawType.AMBIGUOUS
        
        return None
    
    def _find_relevant_line(self, section: str, question: str) -> int:
        """Find the most relevant line number for the question."""
        lines = section.split('\n')
        question_words = set(question.lower().split())
        
        # Remove common words
        common_words = {'what', 'is', 'the', 'are', 'how', 'why', 'when', 'where', 'a', 'an', 'to', 'for'}
        question_words = question_words - common_words
        
        best_line = 1
        best_overlap = 0
        
        for i, line in enumerate(lines, 1):
            line_words = set(line.lower().split())
            overlap = len(question_words & line_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_line = i
        
        return best_line
    
    def _extract_relevant_text(self, section: str, line_num: int) -> str:
        """Extract the relevant text around the line number."""
        lines = section.split('\n')
        if 0 < line_num <= len(lines):
            return lines[line_num - 1].strip()[:100]
        return section[:100] if section else ""
    
    def _generate_recommendation(
        self, 
        flaw_type: FlawType, 
        question: Dict,
        response: SolverResponse
    ) -> str:
        """Generate actionable recommendation for fixing the flaw."""
        
        target = question.get('target', 'this requirement')
        missing = ', '.join(response.missing_info) if response.missing_info else 'missing context'
        
        recommendations = {
            FlawType.AMBIGUOUS: f"Clarify the language. Specify exact values or conditions for: {target}",
            FlawType.MISSING_PREREQ: f"Add prerequisite information. Document should include: {missing}",
            FlawType.IMPLICIT_ASSUMPTION: "Make implicit assumptions explicit. State all required context directly.",
            FlawType.SAFETY_GAP: "CRITICAL: Add safety information. This gap could lead to harm.",
            FlawType.TEMPORAL_GAP: "Add sequence information. Clarify what steps occur between described actions.",
            FlawType.CONTRADICTION: "Resolve contradiction between sections. Ensure consistent information."
        }
        
        return recommendations.get(flaw_type, "Review and improve clarity.")
    
    def summarize_findings(self, findings: List[Finding]) -> Dict:
        """
        Summarize a list of findings by severity and type.
        
        Args:
            findings: List of Finding objects
            
        Returns:
            Summary dictionary with counts by severity and type
        """
        summary = {
            "total": len(findings),
            "by_severity": {},
            "by_type": {}
        }
        
        for severity in Severity:
            count = sum(1 for f in findings if f.severity == severity)
            if count > 0:
                summary["by_severity"][severity.value] = count
        
        for flaw_type in FlawType:
            count = sum(1 for f in findings if f.flaw_type == flaw_type)
            if count > 0:
                summary["by_type"][flaw_type.value] = count

        return summary


if __name__ == "__main__":
    # ponytail: smallest check that fails if the NOT_FOUND dead branch regresses.
    e = Evaluator()
    r = SolverResponse(answer=None, confidence=0.0, status="NOT_FOUND",
                       citations=[], missing_info=[], raw_response="")
    # expected_flaw must be honored across the full set, not collapsed to MISSING_PREREQ
    assert e._classify_flaw({"question": "x", "expected_flaw": "IMPLICIT_ASSUMPTION"}, r) \
        == FlawType.IMPLICIT_ASSUMPTION
    assert e._classify_flaw({"question": "x", "expected_flaw": "TEMPORAL_GAP"}, r) \
        == FlawType.TEMPORAL_GAP
    # safety keyword overrides the tag
    assert e._classify_flaw({"question": "what safety steps?", "expected_flaw": "AMBIGUOUS"}, r) \
        == FlawType.SAFETY_GAP
    # garbage tag falls back
    assert e._classify_flaw({"question": "x", "expected_flaw": "???"}, r) \
        == FlawType.MISSING_PREREQ
    print("OK")
