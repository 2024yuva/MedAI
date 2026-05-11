import numpy as np

from src.ingestion.chunker import dynamic_chunk_text


class StubEmbedding:
    def embed_batch(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            if "fever" in lower or "temperature" in lower:
                vectors.append(np.array([1.0, 0.0], dtype="float32"))
            else:
                vectors.append(np.array([0.0, 1.0], dtype="float32"))
        return vectors


def test_dynamic_chunker_splits_on_semantic_shift() -> None:
    text = (
        "Fever is a common symptom. Temperature may rise with infection. "
        "The patient should rest and hydrate. "
        "Quantum mechanics describes particles at small scales. "
        "Wave functions evolve according to physical laws."
    )

    chunks = dynamic_chunk_text(
        text,
        source_file="doc.pdf",
        page_number=1,
        min_chunk_tokens=1,
        max_chunk_tokens=100,
        similarity_threshold=0.5,
        embedding_service=StubEmbedding(),
    )

    assert len(chunks) >= 2
    assert "fever" in chunks[0].text.lower()
    assert "quantum mechanics" in chunks[-1].text.lower()
