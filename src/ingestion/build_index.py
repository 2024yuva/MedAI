from __future__ import annotations

import argparse

from src.embeddings.service import EmbeddingService
from src.ingestion.pipeline import process_all
from src.retrieval.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="vector_db")
    parser.add_argument(
        "--static-chunking",
        action="store_true",
        help="Use fixed-size chunking instead of semantic dynamic chunking.",
    )
    args = parser.parse_args()

    chunks = process_all(args.data_dir, use_dynamic_chunking=not args.static_chunking)
    if not chunks:
        print("No chunks generated; index not built.")
        return
    emb = EmbeddingService()
    vectors = emb.embed_batch([c.text for c in chunks])
    store = VectorStore()
    store.build_index(vectors, chunks)
    store.save(args.out_dir)
    print(f"Index built with {len(chunks)} chunks at {args.out_dir}")


if __name__ == "__main__":
    main()

