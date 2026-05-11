from __future__ import annotations

import re
from uuid import uuid4
from typing import List, Optional

import numpy as np

from src.models import Chunk


def chunk_text(
    text: str, source_file: str, page_number: int, topic_tag: str = "general", chunk_size: int = 500, overlap: int = 100
) -> List[Chunk]:
    tokens = text.split()
    if not tokens:
        return []
    chunks: List[Chunk] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(tokens), step):
        slice_tokens = tokens[i : i + chunk_size]
        if not slice_tokens:
            continue
        chunks.append(
            Chunk(
                id=str(uuid4()),
                text=" ".join(slice_tokens),
                source_file=source_file,
                page_number=page_number,
                topic_tag=topic_tag,
            )
        )
        if i + chunk_size >= len(tokens):
            break
    return chunks


def _split_into_sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [p.strip() for p in parts if p.strip()]


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def dynamic_chunk_text(
    text: str,
    source_file: str,
    page_number: int,
    topic_tag: str = "general",
    min_chunk_tokens: int = 120,
    max_chunk_tokens: int = 420,
    similarity_threshold: float = 0.72,
    embedding_service: Optional[object] = None,
) -> List[Chunk]:
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    if len(sentences) == 1:
        return [
            Chunk(
                id=str(uuid4()),
                text=sentences[0],
                source_file=source_file,
                page_number=page_number,
                topic_tag=topic_tag,
            )
        ]

    model = embedding_service
    if model is None:
        from src.embeddings.service import EmbeddingService

        model = EmbeddingService()

    sentence_vectors = model.embed_batch(sentences)
    chunks: List[Chunk] = []
    current_sentences: List[str] = []
    current_tokens = 0

    for idx, sentence in enumerate(sentences):
        sentence_tokens = len(sentence.split())
        if not current_sentences:
            current_sentences.append(sentence)
            current_tokens = sentence_tokens
            continue

        prev_vec = sentence_vectors[idx - 1]
        cur_vec = sentence_vectors[idx]
        similarity = _cosine_similarity(prev_vec, cur_vec)

        force_split = current_tokens + sentence_tokens > max_chunk_tokens
        semantic_split = current_tokens >= min_chunk_tokens and similarity < similarity_threshold

        if force_split or semantic_split:
            chunks.append(
                Chunk(
                    id=str(uuid4()),
                    text=" ".join(current_sentences),
                    source_file=source_file,
                    page_number=page_number,
                    topic_tag=topic_tag,
                )
            )
            current_sentences = [sentence]
            current_tokens = sentence_tokens
        else:
            current_sentences.append(sentence)
            current_tokens += sentence_tokens

    if current_sentences:
        chunks.append(
            Chunk(
                id=str(uuid4()),
                text=" ".join(current_sentences),
                source_file=source_file,
                page_number=page_number,
                topic_tag=topic_tag,
            )
        )

    return chunks

