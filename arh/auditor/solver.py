"""
ARH Solver Module

Attempts to answer questions using ONLY the provided document.
Constrained solver that cannot use external knowledge.
"""

import re
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from ..core.agent_wrapper import AgentWrapper


@dataclass
class SolverResponse:
    """Response from the solver's attempt to answer a question."""
    answer: Optional[str]
    confidence: float
    status: str  # FOUND | NOT_FOUND | AMBIGUOUS | PARTIAL
    citations: List[str] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)
    raw_response: str = ""


class Solver:
    """
    Attempts to answer questions using ONLY the provided document.
    Constrained solver that cannot use external knowledge.
    """
    
    def __init__(self, model: AgentWrapper):
        """
        Initialize the solver.
        
        Args:
            model: LLM wrapper for answering questions
        """
        self.model = model
    
    def answer(self, question: str, document: str) -> SolverResponse:
        """
        Answer a question using only the document.
        
        Args:
            question: The question to answer
            document: The document to search for answers
            
        Returns:
            SolverResponse with answer, confidence, and status
        """
        prompt = f"""You are a STRICT documentation validator. You can ONLY use 
information EXPLICITLY stated in the provided document.

CRITICAL RULES:
1. If the answer is NOT EXPLICITLY in the document, respond with STATUS: NOT_FOUND
2. If the answer is AMBIGUOUS (multiple interpretations), respond with STATUS: AMBIGUOUS
3. If you need information NOT in the document, respond with STATUS: NOT_FOUND
4. If you find a partial answer, respond with STATUS: PARTIAL
5. Always cite the SPECIFIC text you're using

DOCUMENT:
{document}

QUESTION: {question}

Respond in this EXACT format:
STATUS: [FOUND|NOT_FOUND|AMBIGUOUS|PARTIAL]
CONFIDENCE: [0-100]
ANSWER: [your answer or "Cannot determine from document"]
CITATION: [exact quote from document, or "N/A"]
MISSING: [what additional info would be needed, or "N/A"]"""

        response = self.model.query(prompt, temperature=0.1)
        
        if response.error:
            return SolverResponse(
                answer=None,
                confidence=0.0,
                status="NOT_FOUND",
                raw_response=f"Error: {response.error}"
            )
        
        return self._parse_response(response.content)
    
    def _parse_response(self, response: str) -> SolverResponse:
        """
        Parse solver response into structured format.

        Tolerant of format drift: case-insensitive, optional markdown bold
        around labels, and ':' or '-' separators. A drifted label must not
        silently drop a finding.

        # ponytail: regex field-grab. JSON-mode prompting is the sturdier
        # upgrade if a provider still slips the format.
        """
        def _field(name: str):
            m = re.search(rf'\**\s*{name}\s*\**\s*[:\-]\s*(.+)', response, re.I)
            return m.group(1).strip().strip('*').strip() if m else None

        # Status: check NOT_FOUND before FOUND ("FOUND" is a substring of it).
        raw_status = (_field('STATUS') or '').upper()
        status = next(
            (s for s in ["NOT_FOUND", "AMBIGUOUS", "PARTIAL", "FOUND"] if s in raw_status),
            "NOT_FOUND",
        )

        confidence = 0.0
        conf_raw = _field('CONFIDENCE')
        if conf_raw:
            m = re.search(r'\d+(?:\.\d+)?', conf_raw)
            if m:
                confidence = max(0.0, min(1.0, float(m.group()) / 100))

        answer = _field('ANSWER')
        cite = _field('CITATION')
        miss = _field('MISSING')
        citations = [cite] if cite and cite.upper() != "N/A" else []
        missing = [miss] if miss and miss.upper() != "N/A" else []

        return SolverResponse(
            answer=answer,
            confidence=confidence,
            status=status,
            citations=citations,
            missing_info=missing,
            raw_response=response,
        )
    
    def answer_simple(self, question: str, document: str) -> SolverResponse:
        """
        DEMO-ONLY — NOT A RELIABILITY SIGNAL. Answers by keyword overlap, not
        comprehension; use the LLM solver for real audits.

        Args:
            question: The question to answer
            document: The document to search
            
        Returns:
            SolverResponse based on keyword matching
        """
        question_words = set(question.lower().split())
        doc_words = set(document.lower().split())
        
        # Check for keyword overlap
        overlap = question_words & doc_words
        
        # Remove common words
        common_words = {'what', 'is', 'the', 'are', 'how', 'why', 'when', 'where', 'a', 'an', 'to', 'for'}
        meaningful_overlap = overlap - common_words
        
        if len(meaningful_overlap) >= 3:
            return SolverResponse(
                answer="Information found in document",
                confidence=0.7,
                status="FOUND",
                citations=[f"Keywords found: {', '.join(list(meaningful_overlap)[:5])}"],
                raw_response="Simple keyword match"
            )
        elif len(meaningful_overlap) >= 1:
            return SolverResponse(
                answer="Partial information may be present",
                confidence=0.4,
                status="PARTIAL",
                missing_info=["More specific information needed"],
                raw_response="Partial keyword match"
            )
        else:
            return SolverResponse(
                answer=None,
                confidence=0.0,
                status="NOT_FOUND",
                missing_info=["No relevant information found in document"],
                raw_response="No keyword match"
            )


if __name__ == "__main__":
    # ponytail: smallest check that fails if tolerant parsing regresses.
    s = Solver.__new__(Solver)  # no model needed for pure parsing
    r = s._parse_response("**STATUS:** NOT_FOUND\n**Confidence** - 30%\nANSWER: n/a")
    assert r.status == "NOT_FOUND", r.status          # not the "FOUND" substring trap
    assert abs(r.confidence - 0.30) < 1e-9, r.confidence
    r2 = s._parse_response("status: found\nconfidence: 90\nCITATION: line 4")
    assert r2.status == "FOUND" and r2.citations == ["line 4"], r2
    print("OK")
