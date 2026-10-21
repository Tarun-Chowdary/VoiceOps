"""Dense (embedding) retrieval with a local multilingual model.

Why e5-small: it embeds Hindi / Hinglish / English into one space, so a Hindi
question can land on an English policy section, which BM25 cannot do.
NOTE: e5 cosine scores cluster in 0.8-0.9, so we only ever use RANKS, never raw thresholds.

EMBED_BACKEND=hash gives a tiny deterministic fake embedder (no model download) used
only for unit tests / CI so the plumbing can be tested without the 0.5 GB model.
"""
import hashlib, os
from functools import lru_cache
import numpy as np
from src import config

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
MODEL_NAME = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
BACKEND = os.getenv("EMBED_BACKEND", "e5")          # "e5" | "hash" (tests only)
_model = None


def _hash_embed(texts, dim=256):
    from src.retrieval.bm25 import tokenize
    out = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for w in tokenize(t):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-9)


def embed(texts: list[str], prefix: str) -> np.ndarray:
    global _model
    if BACKEND == "hash":
        return _hash_embed(texts)
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model.encode([prefix + t for t in texts], normalize_embeddings=True,
                         batch_size=32, show_progress_bar=False)


class DenseIndex:
    def __init__(self, chunks):
        texts = [c.search_text for c in chunks]
        key = hashlib.sha256((BACKEND + MODEL_NAME + "\n".join(texts)).encode()).hexdigest()[:16]
        f = config.ROOT / ".cache" / f"emb_{key}.npy"
        if f.exists():
            self.mat = np.load(f)
        else:
            self.mat = embed(texts, "passage: ")
            f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, self.mat)

    @lru_cache(maxsize=2048)
    def _q(self, query: str):
        return embed([query], "query: ")[0]

    def scores(self, query: str) -> list[float]:
        return (self.mat @ self._q(query)).tolist()
