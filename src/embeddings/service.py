from __future__ import annotations

from typing import List
import numpy as np
import hashlib


class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception:
                self._model = None

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def embed(self, text: str) -> np.ndarray:
        self._load_model()
        if self._model is None:
            out = self._fallback_embed(text)
        else:
            out = self._model.encode([text], convert_to_numpy=True)[0].astype("float32")
        return self._normalize(out)

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[np.ndarray]:
        self._load_model()
        if self._model is None:
            return [self._normalize(self._fallback_embed(t)) for t in texts]
        arr = self._model.encode(texts, batch_size=batch_size, convert_to_numpy=True).astype("float32")
        return [self._normalize(v) for v in arr]

    @staticmethod
    def _fallback_embed(text: str, dim: int = 384) -> np.ndarray:
        vec = np.zeros(dim, dtype="float32")
        tokens = text.lower().split()
        if not tokens:
            return vec
        for tok in tokens:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % dim
            vec[idx] += 1.0
        return vec
