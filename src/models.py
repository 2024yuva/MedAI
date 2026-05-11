from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    id: str
    text: str
    source_file: str
    page_number: int
    topic_tag: str = "general"


@dataclass
class RetrievedContext:
    chunk: Chunk
    similarity_score: float
    rank: int


@dataclass
class GenerationResult:
    raw_text: str
    reasoning_steps: List[str]
    model_used: str
    tokens_generated: int = 0
    latency_ms: int = 0


@dataclass
class SafetyResult:
    safe: bool
    reason: Optional[str] = None
    flagged_patterns: List[str] = field(default_factory=list)

