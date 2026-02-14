"""
Shared LLM judge factory for the benchmarks.

Returns a live LLM judge when an API key is present, else None so callers fall
back to deterministic stubs. The judge is wrapped in a throttle + retry so a
transient rate limit doesn't corrupt a run.
"""

import os
import time


class ThrottledJudge:
    """Duck-typed agent: paces calls and retries on error so a benchmark run
    stays under provider rate limits instead of erroring out."""

    def __init__(self, inner, min_interval: float = 5.0, retries: int = 5):
        self.inner = inner
        self.model = inner.model
        self.min_interval = min_interval
        self.retries = retries
        self._last = 0.0

    def query(self, prompt: str, **kwargs):
        resp = None
        for attempt in range(self.retries):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            resp = self.inner.query(prompt, **kwargs)
            self._last = time.time()
            # A throttled call can return empty content with no error; treat that
            # as retryable so it never silently corrupts a run.
            if not resp.error and resp.content.strip():
                return resp
            time.sleep(min(20.0, 3.0 * (attempt + 1)))  # short backoff; 503s recover fast
        return resp


def get_judge(model: str = None):
    """Return a throttled live judge for the chosen model, else None.

    Model resolution: explicit arg, then ARH_JUDGE_MODEL, then a provider
    default based on whichever API key is set. Throttle spacing is wider for
    Gemini than for OpenAI."""
    from arh.core.agent_wrapper import UniversalWrapper
    model = model or os.getenv("ARH_JUDGE_MODEL")
    if model:
        interval = 1.5 if model.startswith("gemini") else 1.0
        return ThrottledJudge(UniversalWrapper(model=model), min_interval=interval)
    if os.getenv("GEMINI_API_KEY"):
        return ThrottledJudge(UniversalWrapper(model="gemini/gemini-2.5-flash-lite"),
                              min_interval=5.0)
    if os.getenv("OPENAI_API_KEY"):
        return ThrottledJudge(UniversalWrapper(model="gpt-4o-mini"),
                              min_interval=1.0)
    return None


def have_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"))
