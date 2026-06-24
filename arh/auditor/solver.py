"""
ARH Solver Module

Attempts to answer questions using ONLY the provided document.
Constrained solver that cannot use external knowledge.
"""

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
        """Parse solver response into structured format."""
        lines = response.strip().split('\n')
        
        status = "NOT_FOUND"
        confidence = 0.0
        answer = None
        citations = []
        missing = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('STATUS:'):
                status = line.split(':', 1)[1].strip().upper()
                # Normalize status
                if status not in ["FOUND", "NOT_FOUND", "AMBIGUOUS", "PARTIAL"]:
                    status = "NOT_FOUND"
            elif line.startswith('CONFIDENCE:'):
                try:
                    conf_str = line.split(':', 1)[1].strip().replace('%', '')
                    confidence = float(conf_str) / 100
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
                except:
                    confidence = 0.0
            elif line.startswith('ANSWER:'):
                answer = line.split(':', 1)[1].strip()
            elif line.startswith('CITATION:'):
                cite = line.split(':', 1)[1].strip()
                if cite and cite != "N/A":
                    citations.append(cite)
            elif line.startswith('MISSING:'):
                miss = line.split(':', 1)[1].strip()
                if miss and miss != "N/A":
                    missing.append(miss)
        
        return SolverResponse(
            answer=answer,
            confidence=confidence,
            status=status,
            citations=citations,
            missing_info=missing,
            raw_response=response
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
