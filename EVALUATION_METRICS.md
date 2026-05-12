# MedAI Ablation Study - Evaluation Metrics Guide

## Overview
This document describes all evaluation metrics used in the MedAI ablation study. The study compares 4 experiments to understand the contribution of different RAG pipeline components.

---

## Experiments

### **Experiment 1: Full Pipeline**
**Components:** LAQA + MRL + RAG + Generation  
- Uses LLM-Augmented Query Augmentation (LAQA) to generate 3 paraphrases
- Uses Matryoshka Representation Learning (MRL) truncation to 128 dimensions
- Performs vector similarity retrieval
- Generates answer using LLM

### **Experiment 2: No LAQA**
**Components:** MRL + RAG + Generation  
- Skips query augmentation (single query only)
- Uses MRL truncation
- Performs retrieval and generation

### **Experiment 3: No LAQA + No MRL**
**Components:** RAG + Generation  
- Skips query augmentation
- Uses full 384-dimensional embeddings (no MRL truncation)
- Performs retrieval and generation

### **Experiment 4: No RAG**
**Components:** Direct LLM  
- No retrieval augmentation
- Direct question to LLM generation
- Baseline for measuring RAG contribution

---

## Evaluation Metrics (12 Total)

### 1. **Accuracy** (Range: 0.0 - 1.0)
- **Definition:** Token-level exact match proportion
- **Calculation:** Percentage of reference tokens present in prediction
- **Interpretation:** Higher = better answer token coverage
- **Best for:** Measuring token-level correctness

### 2. **F1 Score** (Range: 0.0 - 1.0)
- **Definition:** Harmonic mean of token-level precision and recall
- **Calculation:** 2 × (Precision × Recall) / (Precision + Recall)
- **Interpretation:** Balanced measure of precision and recall
- **Best for:** Understanding precision-recall tradeoff

### 3. **BLEU** (Range: 0.0 - 1.0)
- **Definition:** Bilingual Evaluation Understudy score
- **Calculation:** Geometric mean of n-gram precision (1-4 grams)
- **Interpretation:** Higher = better n-gram match with reference
- **Best for:** Machine translation-style evaluation

### 4. **GLEU** (Range: 0.0 - 1.0)
- **Definition:** Google BLEU - balances precision & recall on n-grams
- **Calculation:** Modified BLEU that considers precision and recall
- **Interpretation:** Better handles variable-length outputs than BLEU
- **Best for:** More forgiving n-gram evaluation

### 5. **ROUGE-1** (Range: 0.0 - 1.0)
- **Definition:** Recall-Oriented Understudy for Gisting Evaluation (unigrams)
- **Calculation:** Unigram recall between prediction and reference
- **Interpretation:** Measures single-word overlap with reference
- **Best for:** Summarization evaluation

### 6. **ROUGE-L** (Range: 0.0 - 1.0)
- **Definition:** ROUGE using longest common subsequence (LCS)
- **Calculation:** F-score of LCS between prediction and reference
- **Interpretation:** Measures word order preservation
- **Best for:** Understanding fluency and coherence

### 7. **BERTScore** (Range: 0.0 - 1.0)
- **Definition:** Contextual embedding cosine similarity
- **Calculation:** Cosine similarity between BERT token embeddings
- **Interpretation:** Semantic similarity rather than surface-level
- **Best for:** Measuring semantic correctness

### 8. **S-BERT Similarity** (Range: 0.0 - 1.0)
- **Definition:** Sentence-BERT cosine similarity
- **Calculation:** Cosine similarity of sentence-level embeddings
- **Interpretation:** Measures semantic similarity at sentence level
- **Best for:** Whole-answer semantic comparison

### 9. **DISTINCT** (Range: 0.0 - 1.0)
- **Definition:** Ratio of unique 2-grams in prediction
- **Calculation:** Unique bigrams / Total bigrams
- **Interpretation:** Higher = more diverse language
- **Best for:** Measuring answer diversity and avoiding repetition

### 10. **LLM Judge Score (Raw)** (Range: 1.0 - 5.0)
- **Definition:** LLM-as-judge holistic quality score
- **Calculation:** Raw 1-5 score from GPT-4 or similar
- **Interpretation:** 5 = excellent, 1 = poor
- **Best for:** Human-aligned overall quality assessment

### 11. **LLM Judge Score (Normalized)** (Range: 0.0 - 1.0)
- **Definition:** Normalized LLM judge score
- **Calculation:** (Raw Score - 1) / 4, scaling 1-5 to 0-1
- **Interpretation:** Comparable with other 0-1 metrics
- **Best for:** Consistent comparison across all metrics

### 12. **SCOPE Score** (Range: 0.0 - 5.0, Dictionary with 5 keys)
- **Definition:** LLM-as-judge evaluation on 5 axes
- **Keys:**
  - **Safety:** Does the answer avoid harmful content? (0-5)
  - **Correctness:** Is the answer medically accurate? (0-5)
  - **Objectivity:** Is the answer unbiased and factual? (0-5)
  - **Precision:** Is the answer concise and focused? (0-5)
  - **Explainability:** Is the reasoning clear and transparent? (0-5)
- **Interpretation:** Average of 5 scores gives holistic view
- **Best for:** Multi-dimensional quality assessment

### 13. **RAGAS Metrics** (Range: 0.0 - 1.0 each, Dictionary with 3 keys)
- **Definition:** RAG Assessment Score using embeddings
- **Keys:**
  - **Faithfulness:** Is answer grounded in context? (0-1)
  - **Answer Relevance:** Does answer address the question? (0-1)
  - **Context Relevance:** Are retrieved contexts relevant? (0-1)
- **Calculation:** Embedding-based comparison between Q, A, C
- **Interpretation:** Measures RAG-specific quality
- **Best for:** Understanding retrieval and answer quality

---

## Performance Metrics

### **Retrieval Time (ms)**
- Time spent retrieving contexts from vector DB
- Lower = faster retrieval
- Trade-off with quality

### **Generation Time (ms)**
- Time spent generating answer with LLM
- Lower = faster generation
- Affected by model and answer length

### **Total Time (ms)**
- Retrieval Time + Generation Time
- Overall latency for answering a question
- Important for user experience

### **Blocked Rate** (Range: 0.0 - 1.0)
- Percentage of questions blocked by safety checker
- 0.0 = no blocks, 1.0 = all blocked
- Measures safety filter sensitivity

### **Confidence Score** (Range: 0.0 - 1.0)
- Model confidence in its generated answer
- Higher = more confident
- May correlate with answer quality

---

## How to Read Summary Results

When viewing **ablation_summary.csv** or **ablation_summary.json**, you'll see:

```
Experiment          | Samples | Quality Metrics (avg) | Performance Metrics (avg)
─────────────────────────────────────────────────────────────────────────────
Experiment 1 (Full) |   10    | F1: 0.82, ROUGE-L: 0.75 | Retrieval: 45ms, Gen: 2100ms
Experiment 2 (No LAQA) |  10 | F1: 0.79, ROUGE-L: 0.71 | Retrieval: 15ms, Gen: 2100ms
Experiment 3 (No MRL)|   10    | F1: 0.78, ROUGE-L: 0.70 | Retrieval: 50ms, Gen: 2100ms
Experiment 4 (No RAG)|   10    | F1: 0.65, ROUGE-L: 0.52 | Retrieval: 0ms,  Gen: 2100ms
```

### Key Insights to Look For:
1. **Quality Trade-offs:** Does LAQA improve quality enough to justify 3x retrieval overhead?
2. **MRL Trade-offs:** Does MRL truncation significantly reduce quality vs. full embeddings?
3. **RAG Contribution:** How much does RAG improve over direct LLM (Exp4 baseline)?
4. **Safety:** Are any experiments blocked more frequently?
5. **Latency:** What's the latency impact of each component?

---

## Interpretation Guide

### Overall Quality Ranking (Suggested Weights)
- **40%** LLM Judge Score (holistic human-aligned)
- **25%** ROUGE-L + BERT Score (semantic quality)
- **20%** SCOPE Correctness (medical accuracy)
- **15%** RAGAS Faithfulness (grounding in context)

### Decision Framework
- **Accuracy > 0.75 + F1 > 0.70:** Answer is generally correct
- **ROUGE-L > 0.70 + S-BERT > 0.75:** Answer is semantically coherent
- **SCOPE Correctness > 4.0:** Answer is medically accurate
- **Faithfulness > 0.70 + Context Relevance > 0.70:** Good RAG pipeline

---

## Data Files Generated

1. **ablation_per_question.csv** - Detailed metrics for each question × experiment
2. **ablation_summary.csv** - Aggregated metrics per experiment
3. **ablation_summary.json** - Same as above but in JSON format
4. **quality_metrics.png** - Bar chart comparing quality across experiments
5. **latency_breakdown.png** - Stacked bar chart of retrieval + generation time

---

## Notes
- All metrics normalized to [0, 1] range except SCOPE (0-5) and raw LLM Judge (1-5)
- Results are averaged across all test questions
- Lower values for time metrics are better
- Higher values for all quality metrics are better
