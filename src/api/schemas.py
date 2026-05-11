from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class SourceReference(BaseModel):
    sourceFile: str
    pageNumber: int
    excerpt: str
    similarityScore: float


class AskResponse(BaseModel):
    finalAnswer: str
    answer: str
    reasoningSteps: list[str]
    sources: list[SourceReference]
    confidenceScore: float
    blocked: bool
    blockReason: str | None = None
