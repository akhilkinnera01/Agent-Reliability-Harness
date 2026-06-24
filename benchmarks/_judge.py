"""
Shared LLM judge factory for the benchmarks.

Returns a live LLM (Gemini by default, OpenAI if its key is set) when an API key
is present, else None so callers fall back to deterministic stubs. Wraps the
model in a throttle + retry so free-tier rate limits don't corrupt a run.
"""

import os
import time


class ThrottledJudge:
    """Duck-typed agent: paces calls and retries on error (e.g. 429 rate limit)
    so benchmark runs stay under free-tier RPM instead of erroring out."""

    def __init__(self, inner, min_interval: float = 13.0, retries: int = 5):
        # Free Gemini tiers allow ~5 req/min -> ~13s spacing keeps us under it.
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
            # A rate-limited Gemini call can return empty content with no error;
            # treat that as retryable so it never silently corrupts a run.
            if not resp.error and resp.content.strip():
                return resp
            time.sleep(min(60.0, 12.0 * (attempt + 1)))  # backoff before retry
        return resp


def get_judge(model: str = "gemini/gemini-2.0-flash"):
    """Return a throttled live judge if an API key is set, else None."""
    if os.getenv("GEMINI_API_KEY"):
        from arh.core.agent_wrapper import UniversalWrapper
        return ThrottledJudge(UniversalWrapper(model=model))
    if os.getenv("OPENAI_API_KEY"):
        from arh.core.agent_wrapper import UniversalWrapper
        return ThrottledJudge(UniversalWrapper(model="gpt-4o-mini"))
    return None


def have_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"))
