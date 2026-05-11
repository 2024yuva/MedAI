from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import AskRequest, AskResponse, SourceReference
from src.embeddings.service import EmbeddingService
from src.generation.service import GenerationService
from src.prompt.builder import PromptBuilder
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import VectorStore
from src.safety.checker import SafetyChecker

LOGGER = logging.getLogger(__name__)


def create_app(vector_db_path: str = "vector_db") -> FastAPI:
    app = FastAPI(title="AI Medical QA MVP")

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    app.state.embedding_service = EmbeddingService()
    app.state.vector_store = VectorStore()
    app.state.prompt_builder = PromptBuilder()
    app.state.generation_service = GenerationService()
    app.state.safety_checker = SafetyChecker()
    app.state.retriever = None

    @app.on_event("startup")
    def startup() -> None:
        app.state.generation_service.log_active_model()
        index_file = Path(vector_db_path) / "index.faiss"
        metadata_file = Path(vector_db_path) / "metadata.json"
        if not index_file.exists() or not metadata_file.exists():
            app.state.index_ready = False
            return
        app.state.vector_store.load(vector_db_path)
        app.state.retriever = Retriever(app.state.embedding_service, app.state.vector_store)
        app.state.index_ready = True

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    def health() -> dict:
        generation_health = app.state.generation_service.health()
        return {
            "status": "ok",
            "activeModel": generation_health["activeModel"],
            "fallbackModel": generation_health["fallbackModel"],
            "ollamaAvailable": generation_health["ollamaReachable"],
        }

    @app.get("/health/generation")
    def generation_health() -> dict:
        return app.state.generation_service.health()

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        total_start = time.perf_counter()
        if not getattr(app.state, "index_ready", False):
            raise HTTPException(status_code=503, detail="Service unavailable - index not built")
        retrieval_start = time.perf_counter()
        contexts, low_conf = app.state.retriever.retrieve(payload.question, k=3)
        retrieval_ms = int((time.perf_counter() - retrieval_start) * 1000)
        prompt = app.state.prompt_builder.build_cot_prompt(payload.question, contexts, low_confidence=low_conf)
        generation_start = time.perf_counter()
        gen = app.state.generation_service.generate(prompt)
        generation_ms = int((time.perf_counter() - generation_start) * 1000)
        total_ms = int((time.perf_counter() - total_start) * 1000)
        LOGGER.info(
            "ask timings retrieval_ms=%s generation_ms=%s total_ms=%s",
            retrieval_ms,
            generation_ms,
            total_ms,
        )
        safety = app.state.safety_checker.check(gen.raw_text)
        sources = [
            SourceReference(
                sourceFile=c.chunk.source_file,
                pageNumber=c.chunk.page_number,
                excerpt=c.chunk.text[:200],
                similarityScore=c.similarity_score,
            )
            for c in contexts
        ]
        confidence = max((c.similarity_score for c in contexts), default=0.0)
        if not safety.safe:
            return AskResponse(
                finalAnswer="Response blocked for safety.",
                answer="Response blocked for safety.",
                reasoningSteps=[],
                sources=sources,
                confidenceScore=confidence,
                blocked=True,
                blockReason=safety.reason,
            )
        answer = gen.raw_text
        if "This is not medical advice." not in answer:
            answer = answer.rstrip() + "\n\nThis is not medical advice."
        return AskResponse(
            finalAnswer=answer,
            answer=answer,
            reasoningSteps=gen.reasoning_steps,
            sources=sources,
            confidenceScore=confidence,
            blocked=False,
            blockReason=None,
        )

    return app


app = create_app()
