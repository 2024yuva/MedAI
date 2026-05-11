import numpy as np

from src.models import Chunk
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore


class StubEmbedding:
    def embed(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype="float32")


def test_retriever_returns_top_k_sorted() -> None:
    store = VectorStore()
    chunks = [
        Chunk(id="1", text="a", source_file="a.pdf", page_number=1),
        Chunk(id="2", text="b", source_file="b.pdf", page_number=1),
    ]
    vectors = [np.array([1.0, 0.0], dtype="float32"), np.array([0.5, 0.0], dtype="float32")]
    store.build_index(vectors, chunks)
    retriever = Retriever(StubEmbedding(), store)
    out, _ = retriever.retrieve("x", k=2)
    assert len(out) == 2
    assert out[0].similarity_score >= out[1].similarity_score

