import numpy as np

from src.embeddings.service import EmbeddingService


def test_embedding_normalize_unit_norm() -> None:
    v = np.array([3.0, 4.0], dtype="float32")
    out = EmbeddingService._normalize(v)
    assert out.shape[0] == 2
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-6

