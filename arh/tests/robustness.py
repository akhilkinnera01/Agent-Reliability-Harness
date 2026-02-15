"""
ARH Robustness Test

Tests agent robustness through prompt perturbations.
A robust agent should produce semantically similar outputs despite
surface-level changes to inputs.

Consistency is judged by semantic similarity (embeddings), not word overlap —
a robust agent can reword its answer and still be correct. Scores are reported
per perturbation type, because typo-fragility and paraphrase-fragility are
different failures with different fixes.
"""

import random
from collections import defaultdict
from typing import List

from ..core.agent_wrapper import AgentWrapper
from ..core.models import TestResult, TestStatus
from ..core.similarity import semantic_similarity


class RobustnessTest:
    """
    Test agent robustness through prompt perturbations.
    A robust agent should produce semantically similar outputs
    despite surface-level changes to inputs.
    """

    def __init__(
        self,
        perturbations: List[str] = None,
        threshold: float = 0.85,
        similarity_threshold: float = 0.85,
    ):
        """
        Initialize robustness test.

        Args:
            perturbations: Types of perturbations to apply
            threshold: Overall pass threshold — fraction of perturbations that
                must stay consistent (0-1)
            similarity_threshold: Per-pair semantic-similarity cutoff above which
                a perturbed answer counts as consistent (0-1). Validated against
                a perturbation benchmark in Phase 3.
        """
        self.perturbations = perturbations or [
            "typo", "rephrase", "case_shift", "noise", "truncate", "paraphrase_llm"
        ]
        self.threshold = threshold
        self.similarity_threshold = similarity_threshold

    def run(self, agent: AgentWrapper, prompts: List[str]) -> TestResult:
        """
        Run robustness test on agent with given prompts.

        Args:
            agent: The agent wrapper to test
            prompts: List of prompts to test with

        Returns:
            TestResult with score, status, and per-perturbation details
        """
        total_tests = 0
        consistent_tests = 0
        failures = []
        # per perturbation type: [total, consistent]
        per_type = defaultdict(lambda: [0, 0])

        for prompt in prompts:
            baseline = agent.query(prompt)
            if baseline.error:
                continue

            for perturb_type in self.perturbations:
                perturbed = self._apply_perturbation(prompt, perturb_type, agent)
                if perturbed == prompt:
                    continue  # perturbation was a no-op (e.g. LLM paraphrase failed)

                response = agent.query(perturbed)
                if response.error:
                    continue

                similarity = semantic_similarity(baseline.content, response.content)
                is_consistent = similarity >= self.similarity_threshold

                total_tests += 1
                per_type[perturb_type][0] += 1
                if is_consistent:
                    consistent_tests += 1
                    per_type[perturb_type][1] += 1
                else:
                    failures.append(
                        f"{perturb_type}: '{prompt[:50]}...' diverged "
                        f"(similarity {similarity:.2f})"
                    )

        score = consistent_tests / total_tests if total_tests > 0 else 0

        per_perturbation = {
            ptype: round(c / t, 3) if t else None
            for ptype, (t, c) in per_type.items()
        }

        return TestResult(
            name="robustness",
            score=score,
            status=TestStatus.PASS if score >= self.threshold else TestStatus.FAIL,
            details={
                "total_tests": total_tests,
                "consistent_tests": consistent_tests,
                "similarity_threshold": self.similarity_threshold,
                "per_perturbation": per_perturbation,
            },
            failures=failures[:10],
            recommendations=self._get_recommendations(score, per_perturbation),
        )

    def _apply_perturbation(
        self, text: str, perturb_type: str, agent: AgentWrapper
    ) -> str:
        """Apply a specific perturbation to the text."""
        if perturb_type == "typo":
            return self._add_typo(text)
        elif perturb_type == "rephrase":
            return self._rephrase(text)
        elif perturb_type == "case_shift":
            return text.upper() if random.random() > 0.5 else text.lower()
        elif perturb_type == "noise":
            return text + " " + "".join(random.choices("asdf ", k=5))
        elif perturb_type == "truncate":
            words = text.split()
            return " ".join(words[:-1]) if len(words) > 1 else text
        elif perturb_type == "paraphrase_llm":
            return self._paraphrase_llm(text, agent)
        return text

    def _add_typo(self, text: str) -> str:
        """Add a realistic typo by swapping two adjacent characters."""
        words = text.split()
        if not words:
            return text
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        if len(word) > 2:
            pos = random.randint(0, len(word) - 2)
            word = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
            words[idx] = word
        return " ".join(words)

    def _rephrase(self, text: str) -> str:
        """Cheap template rephrase (no LLM). See _paraphrase_llm for the real one."""
        if text.endswith("?"):
            return "Can you tell me: " + text
        return text + " Please explain."

    def _paraphrase_llm(self, text: str, agent: AgentWrapper) -> str:
        """
        Ask the agent itself to reword the prompt — a meaning-preserving
        perturbation far harder than typos. Falls back to the original text
        (treated as a no-op skip) if the rewrite fails.
        """
        prompt = (
            "Reword the following request so it means exactly the same thing "
            "but uses different words. Output only the reworded request, "
            f"nothing else:\n\n{text}"
        )
        try:
            resp = agent.query(prompt)
            if resp.error or not resp.content:
                return text
            reworded = resp.content.strip().strip('"')
            return reworded or text
        except Exception:
            return text

    def _get_recommendations(self, score: float, per_perturbation: dict) -> List[str]:
        """Generate recommendations based on which perturbations failed worst."""
        recs = []
        if score < self.threshold:
            recs.append("Consider adding an input normalization layer")
            # Flag the specific weak perturbation types (score below pass threshold).
            for ptype, pscore in per_perturbation.items():
                if pscore is not None and pscore < self.threshold:
                    if ptype == "typo":
                        recs.append("Typo-fragile: add spell-checking / fuzzy matching")
                    elif ptype == "case_shift":
                        recs.append("Case-sensitive: normalize input case")
                    elif ptype == "paraphrase_llm":
                        recs.append("Paraphrase-fragile: answer depends on exact wording")
        return recs
