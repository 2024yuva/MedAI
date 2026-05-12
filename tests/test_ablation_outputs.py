from __future__ import annotations

import csv
import json
from types import SimpleNamespace
from pathlib import Path

from src.ablation import study


class DummyEmbeddingService:
    pass


class DummyVectorStore:
    def load(self, _path: str) -> None:
        return None


class DummyGenerationService:
    pass


class DummyPromptBuilder:
    pass


class DummySafetyChecker:
    pass


def _make_result(experiment: str) -> SimpleNamespace:
    return SimpleNamespace(
        experiment=experiment,
        description=f"{experiment} description",
        final_answer=f"{experiment} answer",
        reasoning_steps=["step 1"],
        sources=[{"excerpt": f"{experiment} context"}],
        confidence_score=0.5,
        retrieval_ms=12,
        generation_ms=34,
        total_ms=46,
        blocked=False,
        block_reason=None,
        augmented_queries=[],
    )


class DummyMetrics:
    def __init__(self) -> None:
        self.accuracy = 0.8
        self.f1_score = 0.7
        self.bleu = 0.6
        self.gleu = 0.5
        self.rouge1 = 0.4
        self.rouge_l = 0.3
        self.bert_score = 0.2
        self.sbert_similarity = 0.1
        self.distinct = 0.9
        self.scope = {
            "safety": 1.0,
            "correctness": 2.0,
            "objectivity": 3.0,
            "precision": 4.0,
            "explainability": 5.0,
        }
        self.ragas = {
            "faithfulness": 0.11,
            "answer_relevance": 0.22,
            "context_relevance": 0.33,
        }
        self.llm_judge_score = 4.0
        self.llm_judge_normalised = 0.8
        self.errors = []


def test_run_study_writes_flat_experiment_outputs(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question": "What is fever?",
                    "reference": "Fever is an elevated body temperature.",
                }
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "reports"

    monkeypatch.setattr(study, "EmbeddingService", DummyEmbeddingService)
    monkeypatch.setattr(study, "VectorStore", DummyVectorStore)
    monkeypatch.setattr(study, "GenerationService", DummyGenerationService)
    monkeypatch.setattr(study, "PromptBuilder", DummyPromptBuilder)
    monkeypatch.setattr(study, "SafetyChecker", DummySafetyChecker)
    monkeypatch.setattr(study, "evaluate_answer", lambda **kwargs: DummyMetrics())
    monkeypatch.setattr(study, "run_exp1_full_pipeline", lambda *args, **kwargs: _make_result("Experiment 1"))
    monkeypatch.setattr(study, "run_exp2_no_laqa", lambda *args, **kwargs: _make_result("Experiment 2"))
    monkeypatch.setattr(study, "run_exp3_no_laqa_no_mrl", lambda *args, **kwargs: _make_result("Experiment 3"))
    monkeypatch.setattr(study, "run_exp4_no_rag", lambda *args, **kwargs: _make_result("Experiment 4"))

    result_dir = study.run_study(dataset_path=dataset_path, output_dir=output_dir, vector_db="unused")

    assert result_dir == output_dir
    assert not list(output_dir.glob("ablation_report_*"))

    expected_files = [
        "experiment_1_per_question.csv",
        "experiment_1_summary.csv",
        "experiment_1_summary.json",
        "experiment_2_per_question.csv",
        "experiment_2_summary.csv",
        "experiment_2_summary.json",
        "experiment_3_per_question.csv",
        "experiment_3_summary.csv",
        "experiment_3_summary.json",
        "experiment_4_per_question.csv",
        "experiment_4_summary.csv",
        "experiment_4_summary.json",
        "ablation_summary.csv",
        "ablation_summary.json",
    ]
    for file_name in expected_files:
        assert (output_dir / file_name).exists()

    with (output_dir / "experiment_1_summary.json").open(encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["experiment"] == "Experiment 1"
    assert summary["samples"] == 1

    with (output_dir / "experiment_1_per_question.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["experiment"] == "Experiment 1"
