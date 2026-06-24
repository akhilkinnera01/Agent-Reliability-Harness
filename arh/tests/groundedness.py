"""
ARH Groundedness Test

DEMO-ONLY — NOT A RELIABILITY SIGNAL. This test infers "hallucination" from
the model's own self-reported confidence and uncertainty phrasing, which
measures politeness, not grounding. Real groundedness requires a source
context + claim-level NLI entailment (Phase 1d rewrite). Do not gate
deployments on this score.
"""

import re
from typing import List, Dict
from ..core.agent_wrapper import AgentWrapper
from ..core.models import TestResult, TestStatus


class GroundednessTest:
    """
    Test agent groundedness by detecting potential hallucinations.
    Uses self-consistency and confidence calibration.
    """
    
    def __init__(self, threshold: float = 0.85):
        """
        Initialize groundedness test.

        Args:
            threshold: Pass threshold (0-1)
        """
        self.threshold = threshold
    
    def run(self, agent: AgentWrapper, prompts: List[str]) -> TestResult:
        """
        Run groundedness test on agent with given prompts.
        
        Args:
            agent: The agent wrapper to test
            prompts: List of prompts to test with
            
        Returns:
            TestResult with score, status, and details
        """
        hallucination_count = 0
        total_tests = 0
        failures = []
        
        for prompt in prompts:
            # Ask the question
            response = agent.query(prompt)
            
            # Skip if initial query failed
            if response.error:
                continue
            
            # Ask for verification/sources
            verification_prompt = f"""
            You previously answered: "{response.content[:200]}"
            
            Rate your confidence (0-100) and explain why.
            If you're not certain, say "I'm not certain because..."
            """
            
            verification = agent.query(verification_prompt)
            
            # Skip if verification failed
            if verification.error:
                continue
            
            # Check for hallucination signals
            is_hallucination = self._detect_hallucination(
                response.content,
                verification.content
            )
            
            total_tests += 1
            if is_hallucination:
                hallucination_count += 1
                failures.append(f"Potential hallucination: '{prompt[:50]}...'")
        
        hallucination_rate = hallucination_count / total_tests if total_tests > 0 else 0
        score = 1 - hallucination_rate
        
        return TestResult(
            name="groundedness",
            score=score,
            status=TestStatus.PASS if score >= self.threshold else TestStatus.FAIL,
            details={
                "hallucination_rate": hallucination_rate,
                "total_tests": total_tests,
                "hallucinations_detected": hallucination_count
            },
            failures=failures[:10],
            recommendations=self._get_recommendations(hallucination_rate)
        )
    
    def _detect_hallucination(self, response: str, verification: str) -> bool:
        """
        Detect if response shows hallucination signals.
        
        Checks for uncertainty signals and low confidence in the verification.
        """
        verification_lower = verification.lower()
        
        # Check for uncertainty signals in verification
        uncertainty_signals = [
            "i'm not certain",
            "i'm not sure",
            "i cannot verify",
            "i don't have",
            "i may have",
            "i might be wrong",
            "i apologize",
            "i cannot confirm"
        ]
        
        for signal in uncertainty_signals:
            if signal in verification_lower:
                return True
        
        # Check for very low confidence
        confidence_match = re.search(r'(\d+)%?\s*(confidence|certain)', verification_lower)
        if confidence_match:
            confidence = int(confidence_match.group(1))
            if confidence < 50:
                return True
        
        return False
    
    def _get_recommendations(self, hallucination_rate: float) -> List[str]:
        """Generate recommendations based on hallucination rate."""
        recs = []
        if hallucination_rate > 0.15:
            recs.append("Implement RAG to ground responses in verified sources")
            recs.append("Add explicit 'I don't know' training")
            recs.append("Consider using a smaller, more constrained model")
        return recs
