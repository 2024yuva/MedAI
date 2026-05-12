"""
Format ablation study results in readable markdown files.
Reads the generated CSV/JSON and creates clean, formatted metric reports.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def _load_json_file(json_file: Path) -> List[Dict[str, Any]]:
    with open(json_file) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON structure in {json_file}")


def load_summary_json(report_dir: Path) -> List[Dict[str, Any]]:
    """Load the ablation_summary.json file or flat experiment summary files."""
    json_file = report_dir / "ablation_summary.json"
    if json_file.exists():
        return _load_json_file(json_file)

    flat_files = sorted(report_dir.glob("experiment_*_summary.json"))
    if flat_files:
        summary: List[Dict[str, Any]] = []
        for file_path in flat_files:
            summary.extend(_load_json_file(file_path))
        return summary

    raise FileNotFoundError(f"Summary JSON not found in {report_dir}")


def load_per_question_csv(report_dir: Path) -> List[Dict[str, str]]:
    """Load the ablation_per_question.csv file or flat experiment CSV files."""
    csv_file = report_dir / "ablation_per_question.csv"
    if csv_file.exists():
        with open(csv_file) as f:
            return list(csv.DictReader(f))

    flat_files = sorted(report_dir.glob("experiment_*_per_question.csv"))
    if flat_files:
        rows: List[Dict[str, str]] = []
        for file_path in flat_files:
            with open(file_path) as f:
                rows.extend(list(csv.DictReader(f)))
        return rows

    raise FileNotFoundError(f"Per-question CSV not found in {report_dir}")


def format_float(val: Any, decimals: int = 3) -> str:
    """Format a value as a float with specified decimals."""
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"


def create_experiment_report(experiment: Dict[str, Any]) -> str:
    """Create a formatted report for a single experiment."""
    lines = []
    exp_name = experiment.get("experiment", "Unknown")
    
    lines.append(f"## {exp_name}\n")
    
    # Basic stats
    lines.append("### Sample Statistics")
    lines.append(f"- **Samples Evaluated:** {int(experiment.get('samples', 0))}")
    lines.append(f"- **Blocked Rate:** {format_float(experiment.get('blockedRate', 0))} ({int(float(experiment.get('blockedRate', 0)) * 100)}%)")
    lines.append(f"- **Average Confidence:** {format_float(experiment.get('confidenceScore', 0))}")
    lines.append("")
    
    # Performance metrics
    lines.append("### Performance Metrics")
    lines.append(f"| Metric | Value |")
    lines.append(f"| --- | --- |")
    lines.append(f"| Retrieval Time | {format_float(experiment.get('retrievalMs', 0), 1)} ms |")
    lines.append(f"| Generation Time | {format_float(experiment.get('generationMs', 0), 1)} ms |")
    lines.append(f"| Total Time | {format_float(experiment.get('totalMs', 0), 1)} ms |")
    lines.append("")
    
    # Quality metrics
    lines.append("### Quality Metrics")
    lines.append(f"| Metric | Score |")
    lines.append(f"| --- | --- |")
    
    quality_metrics = [
        ("accuracy", "Accuracy"),
        ("f1_score", "F1 Score"),
        ("bleu", "BLEU"),
        ("gleu", "GLEU"),
        ("rouge1", "ROUGE-1"),
        ("rouge_l", "ROUGE-L"),
        ("bert_score", "BERTScore"),
        ("sbert_similarity", "S-BERT Similarity"),
        ("distinct", "DISTINCT"),
        ("llm_judge_normalised", "LLM Judge (Norm)"),
    ]
    
    for key, label in quality_metrics:
        value = experiment.get(key, 0)
        lines.append(f"| {label} | {format_float(value, 3)} |")
    lines.append("")
    
    # RAGAS metrics
    lines.append("### RAGAS Metrics (RAG-Specific)")
    lines.append(f"| Metric | Score |")
    lines.append(f"| --- | --- |")
    lines.append(f"| Faithfulness | {format_float(experiment.get('faithfulness', 0), 3)} |")
    lines.append(f"| Answer Relevance | {format_float(experiment.get('answer_relevance', 0), 3)} |")
    lines.append(f"| Context Relevance | {format_float(experiment.get('context_relevance', 0), 3)} |")
    lines.append("")
    
    # Summary assessment
    f1 = float(experiment.get('f1_score', 0))
    rouge_l = float(experiment.get('rouge_l', 0))
    llm_judge = float(experiment.get('llm_judge_normalised', 0))
    
    lines.append("### Quick Assessment")
    if f1 > 0.75 and rouge_l > 0.70:
        lines.append("✅ **Good overall quality** - Strong text similarity metrics")
    elif f1 > 0.65 or rouge_l > 0.60:
        lines.append("⚠️ **Moderate quality** - Some text divergence from reference")
    else:
        lines.append("❌ **Lower quality** - Significant text divergence from reference")
    
    if llm_judge > 0.75:
        lines.append("✅ **Well-evaluated by LLM** - Human-level quality")
    elif llm_judge > 0.5:
        lines.append("⚠️ **Average LLM evaluation** - Room for improvement")
    else:
        lines.append("❌ **Low LLM evaluation** - Quality concerns")
    lines.append("")
    
    return "\n".join(lines)


def create_comparison_table(experiments: List[Dict[str, Any]]) -> str:
    """Create a comparison table across all experiments."""
    lines = []
    lines.append("## Experiment Comparison\n")
    lines.append("### Quality Metrics Comparison")
    lines.append("")
    
    # Header
    header_parts = ["Metric"]
    for exp in experiments:
        header_parts.append(exp.get("experiment", "Exp?")[:15])
    
    lines.append("| " + " | ".join(header_parts) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_parts)) + " |")
    
    # Metrics to compare
    metrics = [
        ("f1_score", "F1 Score"),
        ("rouge_l", "ROUGE-L"),
        ("bert_score", "BERTScore"),
        ("llm_judge_normalised", "LLM Judge"),
        ("faithfulness", "Faithfulness"),
        ("answer_relevance", "Answer Relevance"),
        ("context_relevance", "Context Relevance"),
    ]
    
    for key, label in metrics:
        row_parts = [label]
        for exp in experiments:
            val = format_float(exp.get(key, 0), 3)
            row_parts.append(val)
        lines.append("| " + " | ".join(row_parts) + " |")
    
    lines.append("")
    
    # Performance comparison
    lines.append("### Performance Comparison")
    lines.append("")
    
    header_parts = ["Experiment"]
    for exp in experiments:
        header_parts.append(exp.get("experiment", "Exp?")[:15])
    
    lines.append("| " + " | ".join(header_parts) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_parts)) + " |")
    
    # Retrieval time
    row_parts = ["Retrieval (ms)"]
    for exp in experiments:
        val = format_float(exp.get("retrievalMs", 0), 1)
        row_parts.append(val)
    lines.append("| " + " | ".join(row_parts) + " |")
    
    # Generation time
    row_parts = ["Generation (ms)"]
    for exp in experiments:
        val = format_float(exp.get("generationMs", 0), 1)
        row_parts.append(val)
    lines.append("| " + " | ".join(row_parts) + " |")
    
    # Total time
    row_parts = ["Total (ms)"]
    for exp in experiments:
        val = format_float(exp.get("totalMs", 0), 1)
        row_parts.append(val)
    lines.append("| " + " | ".join(row_parts) + " |")
    
    lines.append("")
    return "\n".join(lines)


def create_insights_section(experiments: List[Dict[str, Any]]) -> str:
    """Create actionable insights from the results."""
    lines = []
    lines.append("## Key Insights\n")
    
    # Find best experiment by different metrics
    best_f1 = max(experiments, key=lambda x: float(x.get("f1_score", 0)))
    best_latency = min(experiments, key=lambda x: float(x.get("totalMs", 999999)))
    best_judge = max(experiments, key=lambda x: float(x.get("llm_judge_normalised", 0)))
    
    lines.append(f"**Best F1 Score:** {best_f1.get('experiment')} ({format_float(best_f1.get('f1_score', 0))})")
    lines.append(f"**Best Latency:** {best_latency.get('experiment')} ({format_float(best_latency.get('totalMs', 0), 1)} ms)")
    lines.append(f"**Best LLM Judge:** {best_judge.get('experiment')} ({format_float(best_judge.get('llm_judge_normalised', 0))})")
    lines.append("")
    
    # Calculate quality vs performance tradeoffs
    lines.append("### Quality vs Performance Trade-offs\n")
    
    # Compare Exp1 vs Exp4 to show RAG contribution
    exp1 = next((e for e in experiments if "1" in e.get("experiment", "")), None)
    exp4 = next((e for e in experiments if "4" in e.get("experiment", "")), None)
    
    if exp1 and exp4:
        f1_improvement = (float(exp1.get("f1_score", 0)) - float(exp4.get("f1_score", 0))) / float(exp4.get("f1_score", 1)) * 100
        latency_diff = float(exp1.get("totalMs", 0)) - float(exp4.get("totalMs", 0))
        lines.append(f"**RAG Contribution (Exp1 vs Exp4):**")
        lines.append(f"- F1 improvement: {f1_improvement:.1f}%")
        lines.append(f"- Latency cost: +{latency_diff:.0f} ms")
        lines.append("")
    
    # Compare Exp1 vs Exp2 to show LAQA value
    exp2 = next((e for e in experiments if "2" in e.get("experiment", "")), None)
    if exp1 and exp2:
        f1_improvement = (float(exp1.get("f1_score", 0)) - float(exp2.get("f1_score", 0))) / float(exp2.get("f1_score", 1)) * 100
        latency_diff = float(exp1.get("totalMs", 0)) - float(exp2.get("totalMs", 0))
        lines.append(f"**LAQA Contribution (Exp1 vs Exp2):**")
        lines.append(f"- F1 improvement: {f1_improvement:.1f}%")
        lines.append(f"- Latency cost: +{latency_diff:.0f} ms")
        lines.append("")
    
    lines.append("### Recommendations\n")
    lines.append("- Choose experiment with best **LLM Judge score** for highest human-aligned quality")
    lines.append("- Consider **latency constraints** when selecting between high-quality and fast options")
    lines.append("- Monitor **faithfulness score** to ensure answers are grounded in retrieved context")
    lines.append("- Check **blocked rate** if safety is critical for your use case")
    
    return "\n".join(lines)


def create_full_report(report_dir: Path) -> str:
    """Create the complete formatted report."""
    summary = load_summary_json(report_dir)
    
    lines = []
    lines.append("# MedAI Ablation Study Results\n")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    lines.append(f"*Report Directory: {report_dir.name}*\n")
    
    # Comparison table
    lines.append(create_comparison_table(summary))
    
    # Individual experiment reports
    lines.append("## Detailed Experiment Results\n")
    for exp in summary:
        lines.append(create_experiment_report(exp))
    
    # Insights
    lines.append(create_insights_section(summary))
    
    return "\n".join(lines)


def main():
    """Main entry point - processes the latest ablation report."""
    reports_dir = Path("reports")

    if not reports_dir.exists():
        print("❌ No reports directory found.")
        return

    has_flat_layout = any(reports_dir.glob("experiment_*_summary.json")) or (reports_dir / "ablation_summary.json").exists()
    if has_flat_layout:
        report_source = reports_dir
    else:
        report_dirs = sorted([d for d in reports_dir.iterdir() if d.is_dir()])
        if not report_dirs:
            print("❌ No ablation reports found in reports/ directory")
            return
        report_source = report_dirs[-1]

    print(f"📊 Processing: {report_source.name}")

    try:
        report_content = create_full_report(report_source)

        output_file = report_source / "FORMATTED_METRICS_REPORT.md"
        output_file.write_text(report_content, encoding="utf-8")

        print(f"✅ Report generated: {output_file}")
        print(f"📝 Open {output_file} to view formatted metrics")

    except Exception as e:
        print(f"❌ Error generating report: {e}")
        raise


if __name__ == "__main__":
    main()
