from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from src.ablation.evaluator import evaluate_answer
from src.ablation.runner import (
    run_exp1_full_pipeline,
    run_exp2_no_laqa,
    run_exp3_no_laqa_no_mrl,
    run_exp4_no_rag,
)
from src.embeddings.service import EmbeddingService
from src.generation.service import GenerationService
from src.prompt.builder import PromptBuilder
from src.retrieval.vector_store import VectorStore
from src.safety.checker import SafetyChecker


EXPERIMENT_ORDER = ["Experiment 1", "Experiment 2", "Experiment 3", "Experiment 4"]
METRICS = [
    "accuracy",
    "f1_score",
    "bleu",
    "gleu",
    "rouge1",
    "rouge_l",
    "bert_score",
    "sbert_similarity",
    "distinct",
    "llm_judge_normalised",
    "faithfulness",
    "answer_relevance",
    "context_relevance",
    "scope_avg",
]


def _load_dataset(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Dataset must be a JSON array")
    items: list[dict[str, str]] = []
    for idx, row in enumerate(raw, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset row {idx} must be an object")
        q = str(row.get("question", "")).strip()
        r = str(row.get("reference", "")).strip()
        if not q or not r:
            raise ValueError(f"Dataset row {idx} must include non-empty question and reference")
        items.append({"question": q, "reference": r})
    return items


def _to_record(
    experiment_result: Any,
    question: str,
    reference: str,
    embedding_service: EmbeddingService,
    generation_service: GenerationService,
) -> dict[str, Any]:
    prediction = experiment_result.final_answer
    contexts = [s.get("excerpt", "") for s in experiment_result.sources]
    eval_result = evaluate_answer(
        question=question,
        prediction=prediction,
        reference=reference,
        contexts=contexts,
        embedding_service=embedding_service,
        generation_service=generation_service,
    )
    scope_values = list(eval_result.scope.values())
    scope_avg = sum(scope_values) / (5 * len(scope_values)) if scope_values else 0.0
    return {
        "experiment": experiment_result.experiment,
        "description": experiment_result.description,
        "question": question,
        "reference": reference,
        "prediction": prediction,
        "blocked": experiment_result.blocked,
        "confidenceScore": experiment_result.confidence_score,
        "retrievalMs": experiment_result.retrieval_ms,
        "generationMs": experiment_result.generation_ms,
        "totalMs": experiment_result.total_ms,
        "accuracy": eval_result.accuracy,
        "f1_score": eval_result.f1_score,
        "bleu": eval_result.bleu,
        "gleu": eval_result.gleu,
        "rouge1": eval_result.rouge1,
        "rouge_l": eval_result.rouge_l,
        "bert_score": eval_result.bert_score,
        "sbert_similarity": eval_result.sbert_similarity,
        "distinct": eval_result.distinct,
        "llm_judge_score": eval_result.llm_judge_score,
        "llm_judge_normalised": eval_result.llm_judge_normalised,
        "faithfulness": eval_result.ragas.get("faithfulness", 0.0),
        "answer_relevance": eval_result.ragas.get("answer_relevance", 0.0),
        "context_relevance": eval_result.ragas.get("context_relevance", 0.0),
        "scope_avg": scope_avg,
    }


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(rec["experiment"], []).append(rec)
    summary: list[dict[str, Any]] = []
    for exp in EXPERIMENT_ORDER:
        rows = grouped.get(exp, [])
        if not rows:
            continue
        out: dict[str, Any] = {
            "experiment": exp,
            "samples": len(rows),
            "blockedRate": mean(1.0 if r["blocked"] else 0.0 for r in rows),
            "confidenceScore": mean(r["confidenceScore"] for r in rows),
            "retrievalMs": mean(r["retrievalMs"] for r in rows),
            "generationMs": mean(r["generationMs"] for r in rows),
            "totalMs": mean(r["totalMs"] for r in rows),
        }
        for metric in METRICS:
            out[metric] = mean(float(r[metric]) for r in rows)
        summary.append(out)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary_rows: list[dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for graph generation. Install it with `pip install matplotlib`.") from exc

    labels = [r["experiment"] for r in summary_rows]

    quality_metrics = ["f1_score", "rouge_l", "bert_score", "llm_judge_normalised", "faithfulness"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(labels))
    width = 0.14
    for i, metric in enumerate(quality_metrics):
        vals = [r[metric] for r in summary_rows]
        offs = [p + (i - 2) * width for p in x]
        ax.bar(offs, vals, width=width, label=metric)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylim(0, 1.0)
    ax.set_title("Ablation Quality Metrics")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "quality_metrics.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    retrieval = [r["retrievalMs"] for r in summary_rows]
    generation = [r["generationMs"] for r in summary_rows]
    ax.bar(labels, retrieval, label="retrievalMs")
    ax.bar(labels, generation, bottom=retrieval, label="generationMs")
    ax.set_title("Latency Breakdown by Experiment (ms)")
    ax.set_ylabel("Milliseconds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "latency_breakdown.png", dpi=180)
    plt.close(fig)


def _experiment_slug(experiment: str) -> str:
    return experiment.lower().replace(" ", "_")


def run_study(dataset_path: Path, output_dir: Path, vector_db: str = "vector_db", k: int = 3) -> Path:
    dataset = _load_dataset(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    vector_store.load(vector_db)
    generation_service = GenerationService()
    prompt_builder = PromptBuilder()
    safety_checker = SafetyChecker()

    experiment_records: dict[str, list[dict[str, Any]]] = {exp: [] for exp in EXPERIMENT_ORDER}
    for item in dataset:
        q = item["question"]
        ref = item["reference"]
        experiments = [
            run_exp1_full_pipeline(q, embedding_service, vector_store, generation_service, prompt_builder, safety_checker, k=k),
            run_exp2_no_laqa(q, embedding_service, vector_store, generation_service, prompt_builder, safety_checker, k=k),
            run_exp3_no_laqa_no_mrl(q, embedding_service, vector_store, generation_service, prompt_builder, safety_checker, k=k),
            run_exp4_no_rag(q, generation_service, safety_checker),
        ]
        for exp_result in experiments:
            record = _to_record(exp_result, q, ref, embedding_service, generation_service)
            experiment_records[record["experiment"]].append(record)

    summary: list[dict[str, Any]] = []
    for experiment in EXPERIMENT_ORDER:
        rows = experiment_records.get(experiment, [])
        if not rows:
            continue
        experiment_summary = _aggregate(rows)
        summary.extend(experiment_summary)

        slug = _experiment_slug(experiment)
        _write_csv(output_dir / f"{slug}_per_question.csv", rows)
        _write_csv(output_dir / f"{slug}_summary.csv", experiment_summary)
        (output_dir / f"{slug}_summary.json").write_text(json.dumps(experiment_summary[0], indent=2), encoding="utf-8")

    _write_csv(output_dir / "ablation_summary.csv", summary)
    (output_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(summary, output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MedAI ablation study with metrics and graphs.")
    parser.add_argument("--dataset", required=True, help="Path to JSON file: [{\"question\":..., \"reference\":...}]")
    parser.add_argument("--output-dir", default="reports", help="Directory where report folder will be created")
    parser.add_argument("--vector-db", default="vector_db", help="Path to vector DB directory")
    parser.add_argument("--k", type=int, default=3, help="Top-k contexts to retrieve")
    args = parser.parse_args()

    report_dir = run_study(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        vector_db=args.vector_db,
        k=args.k,
    )
    print(f"Report generated at: {report_dir}")


if __name__ == "__main__":
    main()
