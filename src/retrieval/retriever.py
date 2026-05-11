from __future__ import annotations

from typing import List, Tuple
import numpy as np

from src.models import RetrievedContext
from src.retrieval.vector_store import VectorStore


class Retriever:
    def __init__(self, embedding_service, vector_store: VectorStore) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 3) -> Tuple[List[RetrievedContext], bool]:
        q = self.embedding_service.embed(query).astype("float32").reshape(1, -1)
        scores, indices = self.vector_store.index.search(q, k)
        contexts: List[RetrievedContext] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(self.vector_store.metadata):
                continue
            contexts.append(
                RetrievedContext(
                    chunk=self.vector_store.metadata[idx],
                    similarity_score=float(score),
                    rank=rank,
                )
            )
        low_confidence = all(c.similarity_score < 0.3 for c in contexts) if contexts else True
        contexts.sort(key=lambda c: c.similarity_score, reverse=True)
        return contexts, low_confidence
