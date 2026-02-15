"""
ARH Proposer Module

Generates adversarial questions designed to expose documentation flaws.
Inspired by Dr. Zero's proposer-solver framework.
"""

import re
from typing import List, Dict
from enum import Enum
from ..core.agent_wrapper import AgentWrapper


class HopComplexity(Enum):
    """Question complexity levels based on reasoning hops required."""
    ONE = 1    # Direct fact retrieval
    TWO = 2    # Cross-reference
    THREE = 3  # Multi-section synthesis
    FOUR = 4   # Edge case reasoning


class Proposer:
    """
    Generates adversarial questions designed to expose documentation flaws.
    Inspired by Dr. Zero's proposer-solver framework.
    """
    
    def __init__(self, model: AgentWrapper):
        """
        Initialize the proposer.
        
        Args:
            model: LLM wrapper for generating questions
        """
        self.model = model
    
    def generate_questions(
        self,
        document: str,
        section: str,
        hop_complexity: List[HopComplexity] = None,
        questions_per_hop: int = 3
    ) -> List[Dict]:
        """
        Generate adversarial questions for a document section.
        
        Args:
            document: Full document text for context
            section: Specific section to audit
            hop_complexity: List of complexity levels to generate
            questions_per_hop: Number of questions per complexity level
            
        Returns:
            List of question dictionaries with question, target, and expected_flaw
        """
        hop_complexity = hop_complexity or [HopComplexity.ONE, HopComplexity.TWO]
        all_questions = []
        
        for hop in hop_complexity:
            prompt = self._build_proposer_prompt(document, section, hop)
            response = self.model.query(prompt, temperature=0.8)
            
            if response.error:
                continue
                
            questions = self._parse_questions(response.content, hop)
            all_questions.extend(questions[:questions_per_hop])
        
        return all_questions
    
    def _build_proposer_prompt(
        self, 
        document: str, 
        section: str, 
        hop: HopComplexity
    ) -> str:
        """Build the proposer prompt based on hop complexity."""
        
        hop_instructions = {
            HopComplexity.ONE: """
Generate questions that test DIRECT FACT RETRIEVAL from this section.
These should be simple questions whose answers should be explicitly stated.
Focus on: specific values, definitions, direct requirements.""",
            
            HopComplexity.TWO: """
Generate questions that require CROSS-REFERENCING within the document.
These should need information from this section plus implied knowledge.
Focus on: relationships, sequences, conditional requirements.""",
            
            HopComplexity.THREE: """
Generate questions that require MULTI-SECTION SYNTHESIS.
These should need combining information from multiple parts.
Focus on: procedures spanning sections, cumulative requirements, dependencies.""",
            
            HopComplexity.FOUR: """
Generate questions about EDGE CASES and FAILURE MODES.
These should probe what happens when things go wrong.
Focus on: exception handling, safety procedures, contingencies."""
        }
        
        return f"""You are an adversarial documentation auditor. Your job is to find 
flaws in documents by generating questions that SHOULD be answerable but likely ARE NOT.

DOCUMENT SECTION:
{section}

FULL DOCUMENT CONTEXT:
{document[:2000]}...

TASK:
{hop_instructions[hop]}

Generate exactly 5 adversarial questions. For each question:
1. It SHOULD be answerable from a complete document
2. It likely EXPOSES a gap, ambiguity, or missing information
3. A real user would reasonably ask this question

Format each question as:
Q1: [question]
TARGET: [what specific info should answer this]
FLAW_IF_MISSING: [AMBIGUOUS|MISSING_PREREQ|IMPLICIT_ASSUMPTION|SAFETY_GAP]

Generate questions:"""

    def _parse_questions(self, response: str, hop: HopComplexity) -> List[Dict]:
        """
        Parse generated questions from model response.

        Tolerant of format drift: accepts ``Q1:``, ``Q1)``, ``1.``, ``Question:``
        for questions and case-insensitive ``TARGET``/``FLAW_IF_MISSING`` (with
        optional markdown) for the fields, so a reworded label does not silently
        drop a finding.

        # ponytail: regex line-matching. JSON-mode prompting is the sturdier
        # upgrade if a provider still slips the format.
        """
        questions = []
        current_q = {}

        q_re = re.compile(r'^\**\s*(?:Q\s*\d*|Question|\d+)\s*\**\s*[:.)]\s*(.+)', re.I)
        target_re = re.compile(r'^\**\s*TARGET\s*\**\s*[:\-]\s*(.+)', re.I)
        flaw_re = re.compile(r'^\**\s*FLAW(?:_IF_MISSING)?\s*\**\s*[:\-]\s*(.+)', re.I)

        def _clean(s: str) -> str:
            return s.strip().strip('*[]').strip()

        for raw in response.splitlines():
            line = re.sub(r'^[-*•]\s+', '', raw.strip())  # drop leading bullet
            # Check field labels before the question pattern (a "1." prefix on a
            # field line would otherwise be misread as a new question).
            mf = flaw_re.match(line)
            mt = target_re.match(line)
            mq = q_re.match(line)
            if mf:
                current_q['expected_flaw'] = _clean(mf.group(1)).upper()
            elif mt:
                current_q['target'] = _clean(mt.group(1))
            elif mq:
                if current_q.get('question'):
                    questions.append(current_q)
                current_q = {'question': _clean(mq.group(1)), 'hop_complexity': hop.value}

        if current_q.get('question'):
            questions.append(current_q)

        return questions

    def is_answerable(self, question: str, document: str) -> bool:
        """
        Answerability gate (soundness check).

        A NOT_FOUND is only a real documentation gap if a *complete* version of
        this document SHOULD answer the question. This filters out-of-scope
        questions — the auditor's biggest false-positive source.

        Fails open (returns True) on judge error so real gaps are never hidden
        by a transient failure.
        """
        prompt = (
            "A document is being audited for gaps. Independent of whether the "
            "current text answers it, decide: SHOULD a complete, high-quality "
            "version of this document reasonably be expected to answer the "
            "QUESTION, given the document's topic and purpose? "
            "Answer with exactly YES or NO.\n\n"
            f"DOCUMENT (excerpt):\n{document[:2000]}\n\nQUESTION: {question}"
        )
        resp = self.model.query(prompt, temperature=0.0)
        if resp.error:
            return True
        return "NO" not in resp.content.strip().upper()[:6]

    def generate_questions_simple(self, section: str, num_questions: int = 5) -> List[Dict]:
        """
        DEMO-ONLY — NOT A RELIABILITY SIGNAL. Returns fixed templated
        questions ignoring document content; use the LLM proposer for real
        audits.

        Args:
            section: Section text to generate questions about
            num_questions: Number of questions to generate
            
        Returns:
            List of simple question dictionaries
        """
        # Common question templates for testing
        templates = [
            {"question": "What are the specific requirements mentioned?", 
             "target": "requirements", "expected_flaw": "AMBIGUOUS"},
            {"question": "What prerequisites are needed before starting?",
             "target": "prerequisites", "expected_flaw": "MISSING_PREREQ"},
            {"question": "What happens if the process fails?",
             "target": "error handling", "expected_flaw": "SAFETY_GAP"},
            {"question": "What are the exact values or thresholds?",
             "target": "specific values", "expected_flaw": "AMBIGUOUS"},
            {"question": "What safety precautions should be taken?",
             "target": "safety info", "expected_flaw": "SAFETY_GAP"},
        ]
        
        return templates[:num_questions]


if __name__ == "__main__":
    # ponytail: smallest check that fails if tolerant question parsing regresses.
    p = Proposer.__new__(Proposer)  # no model needed for pure parsing
    qs = p._parse_questions(
        "**Q1)** What is X?\n- TARGET: the X value\n**FLAW_IF_MISSING** - AMBIGUOUS\n"
        "2. What is Y?",
        HopComplexity.ONE,
    )
    assert len(qs) == 2, qs
    assert qs[0]["question"] == "What is X?", qs[0]
    assert qs[0]["target"] == "the X value", qs[0]
    assert qs[0]["expected_flaw"] == "AMBIGUOUS", qs[0]
    assert qs[1]["question"] == "What is Y?", qs[1]
    print("OK")
