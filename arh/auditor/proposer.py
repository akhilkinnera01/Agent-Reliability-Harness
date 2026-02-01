"""
ARH Proposer Module

Generates adversarial questions designed to expose documentation flaws.
Inspired by Dr. Zero's proposer-solver framework.
"""

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
        """Parse generated questions from model response."""
        questions = []
        current_q = {}
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Q') and ':' in line:
                if current_q and 'question' in current_q:
                    questions.append(current_q)
                current_q = {
                    'question': line.split(':', 1)[1].strip(),
                    'hop_complexity': hop.value
                }
            elif line.startswith('TARGET:'):
                current_q['target'] = line.split(':', 1)[1].strip()
            elif line.startswith('FLAW_IF_MISSING:'):
                current_q['expected_flaw'] = line.split(':', 1)[1].strip()
        
        if current_q and 'question' in current_q:
            questions.append(current_q)
        
        return questions
    
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
