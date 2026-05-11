from __future__ import annotations

import json
from pathlib import Path
from typing import List
import numpy as np
import faiss

from src.models import Chunk


class VectorStore:
    def __init__(self) -> None:
        self.index = None
        self.metadata: List[Chunk] = []

    def build_index(self, embeddings: List[np.ndarray], metadata: List[Chunk]) -> None:
        if not embeddings:
            raise ValueError("No embeddings to index")
        dim = int(embeddings[0].shape[0])
        self.index = faiss.IndexFlatIP(dim)
        matrix = np.vstack(embeddings).astype("float32")
        self.index.add(matrix)
        self.metadata = metadata

    def save(self, path: str) -> None:
        if self.index is None:
            raise ValueError("Index not built")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(out / "index.faiss"))
        meta = [c.__dict__ for c in self.metadata]
        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        base = Path(path)
        self.index = faiss.read_index(str(base / "index.faiss"))
        meta = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
        self.metadata = [Chunk(**m) for m in meta]

