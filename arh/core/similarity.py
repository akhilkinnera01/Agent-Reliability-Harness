"""
Semantic similarity backbone.

One place that answers "how close in meaning are these two strings?" so the
reliability metrics stop relying on word overlap.

Default backend: a small local sentence-transformer (`all-MiniLM-L6-v2`) —
fast, free, no API. If `sentence-transformers` is not installed,
`semantic_similarity` falls back to Jaccard word overlap with a loud one-time
warning so degraded scores are never silent.

    # ponytail: Jaccard fallback only. It is a worse signal, hence the warning;
    # the real fix is `pip install -e .[semantic]`.
"""

import warnings
from functools import lru_cache
from typing import List

import numpy as np

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None          # lazy-loaded SentenceTransformer, or False if unavailable
_warned = False


def _get_model():
    """Lazy-load the embedding model once. Returns None if unavailable."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(_MODEL_NAME)
        except Exception:  # not installed, or download/load failed
            _model = False
    return _model or None


def _warn_degraded():
    global _warned
    if not _warned:
        warnings.warn(
            "sentence-transformers not available — semantic_similarity is "
            "falling back to Jaccard word overlap. Scores are DEGRADED and "
            "not a reliability signal. Install with `pip install -e .[semantic]`.",
            RuntimeWarning,
            stacklevel=2,
        )
        _warned = True


@lru_cache(maxsize=4096)
def _embed_one(text: str) -> tuple:
    """Embed a single string; cached by exact text. Returns a tuple (hashable)."""
    model = _get_model()
    if model is None:
        raise RuntimeError("embedding model unavailable (install arh[semantic])")
    vec = model.encode(text, normalize_embeddings=True)
    return tuple(float(x) for x in vec)


def embed(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts into an (n, d) L2-normalized array.

    Raises RuntimeError if no embedding model is available — clustering and
    other downstream uses need real vectors, so we fail loudly rather than
    hand back a degraded substitute.
    """
    return np.array([_embed_one(t) for t in texts], dtype=np.float32)


def _jaccard(a: str, b: str) -> float:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def semantic_similarity(a: str, b: str) -> float:
    """
    Meaning-level similarity in [0, 1]. 1.0 = same meaning.

    Uses cosine similarity of sentence embeddings; falls back to Jaccard word
    overlap (with a one-time warning) if the embedding model is unavailable.
    """
    if not a or not b:
        return 0.0
    model = _get_model()
    if model is None:
        _warn_degraded()
        return _jaccard(a, b)
    va = np.array(_embed_one(a), dtype=np.float32)
    vb = np.array(_embed_one(b), dtype=np.float32)
    # Embeddings are L2-normalized, so dot product == cosine similarity.
    cos = float(np.dot(va, vb))
    # Map cosine [-1, 1] -> [0, 1] and clamp float noise.
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))


if __name__ == "__main__":
    # ponytail: smallest check that fails if the ordering logic breaks.
    # Works on either backend (embeddings or Jaccard fallback).
    identical = semantic_similarity("the cat sat on the mat",
                                    "the cat sat on the mat")
    paraphrase = semantic_similarity("the cat sat on the mat",
                                     "a feline rested upon the rug")
    unrelated = semantic_similarity("the cat sat on the mat",
                                    "quarterly tax filing deadlines")
    print(f"identical={identical:.3f} paraphrase={paraphrase:.3f} unrelated={unrelated:.3f}")
    assert identical >= 0.99, identical
    assert identical >= paraphrase >= unrelated, (identical, paraphrase, unrelated)
    print("OK")
