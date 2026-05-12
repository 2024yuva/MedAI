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


class AblationExperimentResult(BaseModel):
    experiment: str
    description: str
    finalAnswer: str
    reasoningSteps: list[str]
    sources: list[SourceReference]
    confidenceScore: float
    retrievalMs: int
    generationMs: int
    totalMs: int
    blocked: bool
    blockReason: str | None = None
    augmentedQueries: list[str] = []


class AblationResponse(BaseModel):
    question: str
    results: list[AblationExperimentResult]


class EvaluationRequest(BaseModel):
    question: str = Field(min_length=1)
    reference: str = Field(min_length=1, description="Ground-truth reference answer")


class ScopeScores(BaseModel):
    safety: float = 0.0
    correctness: float = 0.0
    objectivity: float = 0.0
    precision: float = 0.0
    explainability: float = 0.0


class RagasScores(BaseModel):
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_relevance: float = 0.0


class EvaluationResponse(BaseModel):
    question: str
    prediction: str
    reference: str
    accuracy: float
    f1_score: float
    bleu: float
    gleu: float
    rouge1: float
    rouge_l: float
    bert_score: float
    sbert_similarity: float
    distinct: float
    scope: ScopeScores
    ragas: RagasScores
    llm_judge_score: float
    llm_judge_normalised: float
    errors: list[str] = []
