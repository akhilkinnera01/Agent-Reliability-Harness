"""
ARH Groundedness Test

Measures how much of an agent's answer is actually supported by a provided
source context. Groundedness is UNDEFINED without a source — you cannot ask
"is it making this up?" with nothing to check against — so this test is skipped
(not scored 0) when no source is supplied.

Pipeline:
  1. Extract atomic claims from the answer (LLM judge).
  2. Classify each claim against the source via NLI-style entailment:
     ENTAILED / CONTRADICTED / NOT_SUPPORTED (LLM judge, strict rubric).
  3. groundedness = fraction of claims ENTAILED. CONTRADICTED claims are
     flagged critical (a confident falsehood is worse than a gap).

The judge is pluggable. Default is an LLM judge (any AgentWrapper).

    # ponytail: LLM judge by default — no extra deps. Upgrade path is a local
    # cross-encoder NLI model (cross-encoder/nli-deberta-v3-small) for cheaper,
    # more consistent entailment; swap it in via the `judge` arg.

Caveat: an LLM judge is itself imperfect and must be validated against human
labels (Phase 3). Self-judging (judge == agent under test) is allowed but
weaker; pass a separate judge model where possible.
"""

import re
from typing import List, Optional

from ..core.agent_wrapper import AgentWrapper
from ..core.models import TestResult, TestStatus


class GroundednessTest:
    """
    Test agent groundedness via claim extraction + NLI entailment against a
    provided source context.
    """

    def __init__(self, threshold: float = 0.85, judge: Optional[AgentWrapper] = None):
        """
        Initialize groundedness test.

        Args:
            threshold: Pass threshold — minimum fraction of entailed claims (0-1)
            judge: Optional separate LLM used for claim extraction and
                entailment. If None, the agent under test is used (self-judging,
                weaker — see module docstring).
        """
        self.threshold = threshold
        self.judge = judge

    def run(
        self,
        agent: AgentWrapper,
        prompts: List[str],
        sources: Optional[List[str]] = None,
    ) -> TestResult:
        """
        Run groundedness test.

        Args:
            agent: The agent under test (produces the answers)
            prompts: Prompts to answer
            sources: Source context for each prompt to ground against. Must
                align 1:1 with prompts. If omitted, the test is SKIPPED because
                groundedness is undefined without a source.

        Returns:
            TestResult. Status is SKIPPED when no usable sources are provided.
        """
        if not sources or len(sources) != len(prompts):
            return TestResult(
                name="groundedness",
                score=0.0,
                status=TestStatus.SKIPPED,
                details={
                    "reason": "groundedness requires a source context per prompt "
                              "(pass `sources` aligned 1:1 with `prompts`)"
                },
                failures=[],
                recommendations=[
                    "Provide source context (RAG passages, the doc the agent cites) "
                    "to measure grounding."
                ],
            )

        judge = self.judge or agent
        per_prompt_scores = []
        failures = []
        total_claims = 0
        contradictions = 0

        for prompt, source in zip(prompts, sources):
            answer = agent.query(prompt)
            if answer.error or not answer.content.strip():
                continue

            claims = self._extract_claims(judge, answer.content)
            if not claims:
                continue

            entailed = 0
            for claim in claims:
                verdict = self._classify_claim(judge, claim, source)
                total_claims += 1
                if verdict == "ENTAILED":
                    entailed += 1
                elif verdict == "CONTRADICTED":
                    contradictions += 1
                    failures.append(f"CONTRADICTED by source: '{claim[:80]}'")
                else:  # NOT_SUPPORTED
                    failures.append(f"Not supported by source: '{claim[:80]}'")

            per_prompt_scores.append(entailed / len(claims))

        score = sum(per_prompt_scores) / len(per_prompt_scores) if per_prompt_scores else 0.0
        # A contradiction is critical: cap the verdict to FAIL regardless of fraction.
        passed = score >= self.threshold and contradictions == 0

        return TestResult(
            name="groundedness",
            score=score,
            status=TestStatus.PASS if passed else TestStatus.FAIL,
            details={
                "total_claims": total_claims,
                "contradictions": contradictions,
                "prompts_grounded": len(per_prompt_scores),
            },
            failures=failures[:10],
            recommendations=self._get_recommendations(score, contradictions),
        )

    def _extract_claims(self, judge: AgentWrapper, answer: str) -> List[str]:
        """Ask the judge to break the answer into atomic, checkable claims."""
        prompt = (
            "Break the following text into atomic factual claims — one simple, "
            "self-contained statement per line, no numbering, no commentary. "
            "If there are no factual claims, output NONE.\n\n"
            f"TEXT:\n{answer}"
        )
        resp = judge.query(prompt)
        if resp.error:
            return []
        return self._parse_claims(resp.content)

    @staticmethod
    def _parse_claims(text: str) -> List[str]:
        """Tolerant parse: strip bullets/numbering, drop blanks and NONE."""
        claims = []
        for line in text.splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if line and line.upper() != "NONE":
                claims.append(line)
        return claims

    def _classify_claim(self, judge: AgentWrapper, claim: str, source: str) -> str:
        """NLI-style entailment of one claim against the source. Strict rubric."""
        prompt = (
            "You are a strict fact-checker. Given a SOURCE and a CLAIM, decide:\n"
            "- ENTAILED: the source clearly supports the claim.\n"
            "- CONTRADICTED: the source clearly states the opposite.\n"
            "- NOT_SUPPORTED: the source neither supports nor contradicts it.\n"
            "Answer with exactly one word: ENTAILED, CONTRADICTED, or NOT_SUPPORTED.\n\n"
            f"SOURCE:\n{source}\n\nCLAIM:\n{claim}"
        )
        resp = judge.query(prompt)
        if resp.error:
            return "NOT_SUPPORTED"
        return self._parse_verdict(resp.content)

    @staticmethod
    def _parse_verdict(text: str) -> str:
        """Tolerant verdict parse — find the rubric word anywhere in the reply."""
        upper = text.upper()
        if "CONTRADICT" in upper:
            return "CONTRADICTED"
        if "ENTAIL" in upper:
            return "ENTAILED"
        return "NOT_SUPPORTED"

    def _get_recommendations(self, score: float, contradictions: int) -> List[str]:
        """Generate recommendations based on grounding results."""
        recs = []
        if contradictions:
            recs.append("Critical: answer contradicts its source — investigate before deploy")
        if score < self.threshold:
            recs.append("Ground responses in retrieved sources (RAG) and cite them")
            recs.append("Constrain the model to answer only from provided context")
        return recs


if __name__ == "__main__":
    # ponytail: smallest checks that fail if parsing or aggregation breaks.
    assert GroundednessTest._parse_verdict("The answer is ENTAILED.") == "ENTAILED"
    assert GroundednessTest._parse_verdict("clearly CONTRADICTED here") == "CONTRADICTED"
    assert GroundednessTest._parse_verdict("hmm, unclear") == "NOT_SUPPORTED"
    assert GroundednessTest._parse_claims("1. cats purr\n- dogs bark\nNONE\n") == \
        ["cats purr", "dogs bark"]

    # Scripted judge to exercise run() aggregation without a real LLM.
    from ..core.models import AgentResponse

    class _StubJudge(AgentWrapper):
        def __init__(self):
            super().__init__(endpoint="stub://", model="stub")
        def query(self, prompt: str, **kw) -> AgentResponse:
            if "Break the following" in prompt:
                return AgentResponse(content="claim A\nclaim B", latency_ms=0, model="stub")
            return AgentResponse(content="ENTAILED", latency_ms=0, model="stub")

    t = GroundednessTest(judge=_StubJudge())
    skipped = t.run(_StubJudge(), ["q"], sources=None)
    assert skipped.status == TestStatus.SKIPPED, skipped.status
    scored = t.run(_StubJudge(), ["q"], sources=["some source"])
    assert scored.score == 1.0 and scored.status == TestStatus.PASS, scored
    print("OK")
