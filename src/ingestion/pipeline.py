from __future__ import annotations

from typing import List

from src.models import Chunk
from src.ingestion.loader import load_pdfs
from src.ingestion.cleaner import clean_text
from src.ingestion.chunker import chunk_text, dynamic_chunk_text


def process_all(data_dir: str, use_dynamic_chunking: bool = True) -> List[Chunk]:
    docs = load_pdfs(data_dir)
    all_chunks: List[Chunk] = []
    for doc in docs:
        for idx, page_text in enumerate(doc.pages, start=1):
            cleaned = clean_text(page_text)
            if use_dynamic_chunking:
                all_chunks.extend(dynamic_chunk_text(cleaned, source_file=doc.source_file, page_number=idx))
            else:
                all_chunks.extend(chunk_text(cleaned, source_file=doc.source_file, page_number=idx))
    return all_chunks

