"""
Shared LLM judge factory for the benchmarks.

Returns a live LLM (Gemini by default, OpenAI if its key is set) when an API key
is present, else None so callers fall back to deterministic stubs. Keeps the
benchmark code itself provider-agnostic.
"""

import os
from typing import Optional


def get_judge(model: str = "gemini/gemini-2.5-flash"):
    """Return a live UniversalWrapper if an API key is set, else None."""
    if os.getenv("GEMINI_API_KEY"):
        from arh.core.agent_wrapper import UniversalWrapper
        return UniversalWrapper(model=model)
    if os.getenv("OPENAI_API_KEY"):
        from arh.core.agent_wrapper import UniversalWrapper
        return UniversalWrapper(model="gpt-4o-mini")
    return None


def have_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"))
