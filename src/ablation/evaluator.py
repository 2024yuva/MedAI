"""
Evaluation metrics for ablation study — Experiment 1 (Full Pipeline).

Metrics:
  1.  Accuracy        — token-level exact match proportion
  2.  F1 Score        — harmonic mean of token precision & recall
  3.  BLEU            — n-gram precision (1–4), geometric mean
  4.  GLEU            — Google-BLEU: balances precision & recall on n-grams
  5.  ROUGE-1         — unigram recall between prediction and reference
  6.  ROUGE-L         — longest common subsequence F1
  7.  BERTScore       — contextual embedding cosine similarity (uses project's SentenceTransformer)
  8.  S-BERT          — sentence-level cosine similarity via SBERT
  9.  DISTINCT        — ratio of unique 2-grams in prediction (diversity)
  10. SCOPE           — LLM-as-judge on 5 axes: Safety, Correctness, Objectivity, Precision, Explainability
  11. RAGAS           — Faithfulness, Answer Relevance, Context Relevance via embeddings
  12. LLM-as-a-Judge  — single holistic score (1–5) from LLM

All metrics return float values in [0, 1] except SCOPE (dict of 5 scores 0–5)
and LLM-as-a-Judge (1–5 raw + normalised 0–1).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationMetrics:
    accuracy: float = 0.0
    f1_score: float = 0.0
    bleu: float = 0.0
    gleu: float = 0.0
    rouge1: float = 0.0
    rouge_l: float = 0.0
    bert_score: float = 0.0
    sbert_similarity: float = 0.0
    distinct: float = 0.0
    scope: Dict[str, float] = field(default_factory=dict)
    ragas: Dict[str, float] = field(default_factory=dict)
    llm_judge_score: float = 0.0          # raw 1–5
    llm_judge_normalised: float = 0.0     # 0–1
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "f1_score": round(self.f1_score, 4),
            "bleu": round(self.bleu, 4),
            "gleu": round(self.gleu, 4),
            "rouge1": round(self.rouge1, 4),
            "rouge_l": round(self.rouge_l, 4),
            "bert_score": round(self.bert_score, 4),
            "sbert_similarity": round(self.sbert_similarity, 4),
            "distinct": round(self.distinct, 4),
            "scope": {k: round(v, 2) for k, v in self.scope.items()},
            "ragas": {k: round(v, 4) for k, v in self.ragas.items()},
            "llm_judge_score": round(self.llm_judge_score, 2),
            "llm_judge_normalised": round(self.llm_judge_normalised, 4),
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Tokenisation helper
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


# ---------------------------------------------------------------------------
# 1. Accuracy — proportion of reference tokens present in prediction
# ---------------------------------------------------------------------------

def accuracy(prediction: str, reference: str) -> float:
    pred_tokens = set(_tokenize(prediction))
    ref_tokens = set(_tokenize(reference))
    if not ref_tokens:
        return 0.0
    return len(pred_tokens & ref_tokens) / len(ref_tokens)


# ---------------------------------------------------------------------------
# 2. F1 Score — token-level precision & recall harmonic mean
# ---------------------------------------------------------------------------

def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    common = sum((pred_counter & ref_counter).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# 3. BLEU — modified n-gram precision, geometric mean of 1–4 grams
# ---------------------------------------------------------------------------

def bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else math.exp(1 - len(ref_tokens) / len(pred_tokens))

    log_sum = 0.0
    for n in range(1, max_n + 1):
        pred_ng = _ngrams(pred_tokens, n)
        ref_ng = _ngrams(ref_tokens, n)
        clipped = sum(min(cnt, ref_ng[ng]) for ng, cnt in pred_ng.items())
        total = max(len(pred_tokens) - n + 1, 0)
        if total == 0 or clipped == 0:
            return 0.0
        log_sum += math.log(clipped / total)

    return bp * math.exp(log_sum / max_n)


# ---------------------------------------------------------------------------
# 4. GLEU — Google-BLEU: min(precision, recall) on n-gram matches
# ---------------------------------------------------------------------------

def gleu(prediction: str, reference: str, max_n: int = 4) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        pred_ng = _ngrams(pred_tokens, n)
        ref_ng = _ngrams(ref_tokens, n)
        matches = sum(min(cnt, ref_ng[ng]) for ng, cnt in pred_ng.items())
        pred_total = max(len(pred_tokens) - n + 1, 0)
        ref_total = max(len(ref_tokens) - n + 1, 0)
        if pred_total == 0 or ref_total == 0:
            continue
        precision = matches / pred_total
        recall = matches / ref_total
        scores.append(min(precision, recall))

    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# 5. ROUGE-1 — unigram F1
# ---------------------------------------------------------------------------

def rouge1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    overlap = sum((pred_counter & ref_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# 6. ROUGE-L — longest common subsequence F1
# ---------------------------------------------------------------------------

def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS via DP
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / m
    recall = lcs / n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# 7. BERTScore — token-level contextual similarity via SentenceTransformer
#    (uses the project's existing EmbeddingService as a proxy)
# ---------------------------------------------------------------------------

def bert_score(prediction: str, reference: str, embedding_service) -> float:
    try:
        # Split into sentences and embed each; take mean cosine similarity
        pred_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", prediction) if s.strip()]
        ref_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", reference) if s.strip()]
        if not pred_sents or not ref_sents:
            return 0.0

        pred_vecs = np.array(embedding_service.embed_batch(pred_sents))
        ref_vecs = np.array(embedding_service.embed_batch(ref_sents))

        # Precision: for each pred sentence, max cosine sim to any ref sentence
        precision_scores = [float(np.max(pred_vecs[i] @ ref_vecs.T)) for i in range(len(pred_vecs))]
        # Recall: for each ref sentence, max cosine sim to any pred sentence
        recall_scores = [float(np.max(ref_vecs[i] @ pred_vecs.T)) for i in range(len(ref_vecs))]

        p = sum(precision_scores) / len(precision_scores)
        r = sum(recall_scores) / len(recall_scores)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
    except Exception as e:
        return 0.0


# ---------------------------------------------------------------------------
# 8. S-BERT — sentence-level cosine similarity
# ---------------------------------------------------------------------------

def sbert_similarity(prediction: str, reference: str, embedding_service) -> float:
    try:
        pred_vec = embedding_service.embed(prediction)
        ref_vec = embedding_service.embed(reference)
        return float(np.clip(np.dot(pred_vec, ref_vec), 0.0, 1.0))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# 9. DISTINCT — ratio of unique 2-grams (diversity measure)
# ---------------------------------------------------------------------------

def distinct(prediction: str, n: int = 2) -> float:
    tokens = _tokenize(prediction)
    if len(tokens) < n:
        return 0.0
    all_ngrams = list(_ngrams(tokens, n).keys())
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


# ---------------------------------------------------------------------------
# 10. SCOPE — LLM-as-judge on 5 axes (0–5 each)
# ---------------------------------------------------------------------------

def scope_framework(
    question: str,
    prediction: str,
    generation_service,
) -> Dict[str, float]:
    prompt = (
        "You are an expert medical evaluator. Score the following answer on each axis from 0 to 5.\n"
        "Output ONLY a JSON object with keys: safety, correctness, objectivity, precision, explainability.\n"
        "No explanation, just the JSON.\n\n"
        f"Question: {question}\n\nAnswer: {prediction}\n\n"
        "JSON scores:"
    )
    try:
        result = generation_service.generate(prompt)
        raw = result.raw_text
        # Extract JSON from response
        match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in SCOPE response")
        import json
        scores = json.loads(match.group())
        keys = ["safety", "correctness", "objectivity", "precision", "explainability"]
        return {k: float(scores.get(k, 0)) for k in keys}
    except Exception as e:
        return {"safety": 0, "correctness": 0, "objectivity": 0, "precision": 0, "explainability": 0}


# ---------------------------------------------------------------------------
# 11. RAGAS — Faithfulness, Answer Relevance, Context Relevance
#     Computed via embedding cosine similarity (no external RAGAS library needed)
# ---------------------------------------------------------------------------

def ragas_metrics(
    question: str,
    prediction: str,
    contexts: List[str],
    embedding_service,
) -> Dict[str, float]:
    try:
        context_text = " ".join(contexts)

        # Faithfulness: how much of the answer is grounded in the context
        faithfulness = sbert_similarity(prediction, context_text, embedding_service)

        # Answer Relevance: how relevant the answer is to the question
        answer_relevance = sbert_similarity(prediction, question, embedding_service)

        # Context Relevance: how relevant the retrieved context is to the question
        context_relevance = sbert_similarity(context_text, question, embedding_service)

        return {
            "faithfulness": round(float(faithfulness), 4),
            "answer_relevance": round(float(answer_relevance), 4),
            "context_relevance": round(float(context_relevance), 4),
        }
    except Exception:
        return {"faithfulness": 0.0, "answer_relevance": 0.0, "context_relevance": 0.0}


# ---------------------------------------------------------------------------
# 12. LLM-as-a-Judge — holistic score 1–5
# ---------------------------------------------------------------------------

def llm_as_judge(
    question: str,
    prediction: str,
    generation_service,
) -> tuple[float, float]:
    """Returns (raw_score 1–5, normalised 0–1)."""
    prompt = (
        "You are a medical QA evaluator. Rate the following answer on correctness, "
        "completeness, relevance, and safety on a scale of 1 to 5.\n"
        "Output ONLY a single integer between 1 and 5. Nothing else.\n\n"
        f"Question: {question}\n\nAnswer: {prediction}\n\nScore:"
    )
    try:
        result = generation_service.generate(prompt)
        raw = result.raw_text.strip()
        match = re.search(r"\b([1-5])\b", raw)
        if not match:
            raise ValueError(f"No score found in: {raw}")
        score = float(match.group(1))
        return score, (score - 1) / 4.0   # normalise to [0, 1]
    except Exception:
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def evaluate_answer(
    question: str,
    prediction: str,
    reference: str,
    contexts: List[str],
    embedding_service,
    generation_service,
) -> EvaluationMetrics:
    """Compute all 12 metrics for one question/prediction pair."""
    m = EvaluationMetrics()

    # --- Lexical metrics ---
    m.accuracy = accuracy(prediction, reference)
    m.f1_score = f1_score(prediction, reference)
    m.bleu = bleu(prediction, reference)
    m.gleu = gleu(prediction, reference)
    m.rouge1 = rouge1(prediction, reference)
    m.rouge_l = rouge_l(prediction, reference)

    # --- Diversity ---
    m.distinct = distinct(prediction)

    # --- Semantic metrics ---
    m.bert_score = bert_score(prediction, reference, embedding_service)
    m.sbert_similarity = sbert_similarity(prediction, reference, embedding_service)

    # --- RAGAS ---
    m.ragas = ragas_metrics(question, prediction, contexts, embedding_service)

    # --- LLM-based metrics ---
    m.scope = scope_framework(question, prediction, generation_service)
    m.llm_judge_score, m.llm_judge_normalised = llm_as_judge(question, prediction, generation_service)

    return m


def evaluate_experiment1(
    question: str,
    prediction: str,
    reference: str,
    contexts: List[str],
    embedding_service,
    generation_service,
) -> EvaluationMetrics:
    """
    Backward-compatible wrapper for existing code paths that score Exp 1.
    """
    return evaluate_answer(
        question=question,
        prediction=prediction,
        reference=reference,
        contexts=contexts,
        embedding_service=embedding_service,
        generation_service=generation_service,
    )
