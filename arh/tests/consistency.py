"""
ARH Consistency Test

Tests agent consistency by querying the same prompt multiple times.
A consistent agent produces semantically EQUIVALENT responses — it may reword,
but it should not change its mind.

Consistency is scored with semantic entropy: the N samples are clustered by
meaning (embed -> greedy threshold cluster), and the score is 1 minus the
normalized entropy of the cluster sizes. One cluster = perfectly consistent
(score 1); every sample in its own cluster = maximally inconsistent (score 0).
This is far more faithful than averaging pairwise word-set distance, which
punishes harmless rewording.
"""

import math
from typing import List

import numpy as np

from ..core.agent_wrapper import AgentWrapper
from ..core.models import TestResult, TestStatus
from ..core.similarity import semantic_similarity


class ConsistencyTest:
    """
    Test agent consistency by querying the same prompt multiple times.
    A consistent agent produces semantically equivalent responses.
    """

    def __init__(
        self,
        samples: int = 5,
        threshold: float = 0.90,
        temperature: float = 0.7,
        cluster_threshold: float = 0.85,
    ):
        """
        Initialize consistency test.

        Args:
            samples: Number of times to query each prompt
            threshold: Overall pass threshold (0-1)
            temperature: Temperature to use for queries
            cluster_threshold: Semantic-similarity cutoff above which two
                samples are treated as the same answer (0-1)
        """
        self.samples = samples
        self.threshold = threshold
        self.temperature = temperature
        self.cluster_threshold = cluster_threshold

    def run(self, agent: AgentWrapper, prompts: List[str]) -> TestResult:
        """
        Run consistency test on agent with given prompts.

        Args:
            agent: The agent wrapper to test
            prompts: List of prompts to test with

        Returns:
            TestResult with score, status, and details
        """
        per_prompt_scores = []
        failures = []

        for prompt in prompts:
            responses = [
                agent.query(prompt, temperature=self.temperature)
                for _ in range(self.samples)
            ]
            contents = [r.content for r in responses if not r.error]

            if len(contents) < 2:
                continue  # not enough successful responses to judge consistency

            clusters = self._cluster(contents)
            prompt_score = 1.0 - self._normalized_entropy(clusters, len(contents))
            per_prompt_scores.append(prompt_score)

            if prompt_score < self.threshold:
                failures.append(
                    f"Inconsistent ({len(clusters)} distinct answers): "
                    f"'{prompt[:50]}...'"
                )

        score = float(np.mean(per_prompt_scores)) if per_prompt_scores else 0.0

        return TestResult(
            name="consistency",
            score=score,
            status=TestStatus.PASS if score >= self.threshold else TestStatus.FAIL,
            details={
                "samples_per_prompt": self.samples,
                "prompts_tested": len(prompts),
                "cluster_threshold": self.cluster_threshold,
            },
            failures=failures[:10],
            recommendations=self._get_recommendations(score),
        )

    def _cluster(self, contents: List[str]) -> List[List[str]]:
        """
        Greedy semantic clustering: each sample joins the first cluster whose
        representative it is equivalent to, else starts a new cluster.

        # ponytail: greedy single-pass clustering, O(n*k). n is tiny (samples
        # per prompt), so agglomerative clustering would be over-engineering.
        """
        clusters: List[List[str]] = []
        for text in contents:
            for cluster in clusters:
                if semantic_similarity(text, cluster[0]) >= self.cluster_threshold:
                    cluster.append(text)
                    break
            else:
                clusters.append([text])
        return clusters

    def _normalized_entropy(self, clusters: List[List[str]], n: int) -> float:
        """
        Shannon entropy of cluster sizes, normalized to [0, 1] by log(n).

        0.0 = all samples in one cluster (consistent);
        1.0 = every sample in its own cluster (maximally inconsistent).
        """
        if n <= 1:
            return 0.0
        entropy = 0.0
        for cluster in clusters:
            p = len(cluster) / n
            entropy -= p * math.log(p)
        return entropy / math.log(n)

    def _get_recommendations(self, score: float) -> List[str]:
        """Generate recommendations based on score."""
        recs = []
        if score < self.threshold:
            recs.append("Consider lowering temperature for more deterministic outputs")
            recs.append("Implement response caching for identical queries")
        return recs


if __name__ == "__main__":
    # ponytail: smallest check that fails if clustering/entropy breaks.
    t = ConsistencyTest()
    same = t._cluster(["the sky is blue"] * 4)
    assert t._normalized_entropy(same, 4) == 0.0
    diff = t._cluster([
        "the sky is blue", "tax returns are due", "cats enjoy sleeping",
        "the engine overheated",
    ])
    assert t._normalized_entropy(diff, 4) > 0.99
    print("OK")
