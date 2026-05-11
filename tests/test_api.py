from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.models import Chunk, GenerationResult
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore


class StubEmbedding:
    def embed(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype="float32")


def _seed_index(path: Path) -> None:
    store = VectorStore()
    chunk = Chunk(id="1", text="context text about fever", source_file="doc.pdf", page_number=2)
    store.build_index([np.array([1.0, 0.0], dtype="float32")], [chunk])
    store.save(str(path))


def test_ask_503_when_index_missing(tmp_path: Path) -> None:
    app = create_app(str(tmp_path / "missing"))
    client = TestClient(app)
    res = client.post("/ask", json={"question": "What is fever?"})
    assert res.status_code == 503


def test_ask_422_for_invalid_payload(tmp_path: Path) -> None:
    index_dir = tmp_path / "vector_db"
    _seed_index(index_dir)
    app = create_app(str(index_dir))
    app.state.vector_store.load(str(index_dir))
    app.state.retriever = Retriever(StubEmbedding(), app.state.vector_store)
    app.state.index_ready = True
    client = TestClient(app)
    res = client.post("/ask", json={"question": ""})
    assert res.status_code == 422


def test_health_contract(tmp_path: Path) -> None:
    app = create_app(str(tmp_path / "missing"))
    app.state.generation_service.health = lambda: {
        "activeModel": "phi3",
        "fallbackModel": None,
        "ollamaReachable": True,
    }
    client = TestClient(app)

    res = client.get("/health")

    assert res.status_code == 200
    assert res.json() == {
        "status": "ok",
        "activeModel": "phi3",
        "fallbackModel": None,
        "ollamaAvailable": True,
    }


def test_ask_success_and_disclaimer(tmp_path: Path) -> None:
    index_dir = tmp_path / "vector_db"
    _seed_index(index_dir)
    app = create_app(str(index_dir))

    app.state.vector_store.load(str(index_dir))
    app.state.embedding_service = StubEmbedding()
    app.state.retriever = Retriever(app.state.embedding_service, app.state.vector_store)
    app.state.index_ready = True
    app.state.generation_service.generate = lambda _prompt: GenerationResult(
        raw_text="1. step\n2. step\n3. step\n4. step\n5. step",
        reasoning_steps=["step1", "step2", "step3", "step4", "step5"],
        model_used="phi3",
    )

    client = TestClient(app)
    res = client.post("/ask", json={"question": "What is fever?"})
    assert res.status_code == 200
    body = res.json()
    assert body["blocked"] is False
    assert len(body["sources"]) >= 1
    assert body["finalAnswer"]
    assert "This is not medical advice." in body["answer"]


def test_ask_blocked_response(tmp_path: Path) -> None:
    index_dir = tmp_path / "vector_db"
    _seed_index(index_dir)
    app = create_app(str(index_dir))
    app.state.vector_store.load(str(index_dir))
    app.state.embedding_service = StubEmbedding()
    app.state.retriever = Retriever(app.state.embedding_service, app.state.vector_store)
    app.state.index_ready = True
    app.state.generation_service.generate = lambda _prompt: GenerationResult(
        raw_text="Take 500mg ibuprofen now.",
        reasoning_steps=["a", "b", "c", "d", "e"],
        model_used="phi3",
    )
    client = TestClient(app)
    res = client.post("/ask", json={"question": "Pain?"})
    assert res.status_code == 200
    body = res.json()
    assert body["blocked"] is True
    assert body["blockReason"]
