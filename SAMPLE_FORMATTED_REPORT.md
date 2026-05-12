# MedAI Ablation Study Results

*This is a SAMPLE showing what your final report will look like*  
*The actual values will come from your ablation study results*

---

## Experiment Comparison

### Quality Metrics Comparison

| Metric | Experiment 1 | Experiment 2 | Experiment 3 | Experiment 4 |
| --- | --- | --- | --- | --- |
| F1 Score | 0.821 | 0.789 | 0.776 | 0.654 |
| ROUGE-L | 0.745 | 0.712 | 0.698 | 0.523 |
| BERTScore | 0.834 | 0.805 | 0.791 | 0.689 |
| LLM Judge | 0.854 | 0.821 | 0.809 | 0.675 |
| Faithfulness | 0.823 | 0.791 | 0.775 | 0.521 |
| Answer Relevance | 0.879 | 0.856 | 0.843 | 0.712 |
| Context Relevance | 0.801 | 0.768 | 0.754 | 0.134 |

### Performance Comparison

| Experiment | Retrieval (ms) | Generation (ms) | Total (ms) |
| --- | --- | --- | --- |
| Experiment 1 | 123.4 | 2087.3 | 2210.7 |
| Experiment 2 | 41.2 | 2089.1 | 2130.3 |
| Experiment 3 | 89.7 | 2091.5 | 2181.2 |
| Experiment 4 | 0.0 | 2084.6 | 2084.6 |

---

## Detailed Experiment Results

## Experiment 1 (Full Pipeline)

### Sample Statistics
- **Samples Evaluated:** 10
- **Blocked Rate:** 0.100 (10%)
- **Average Confidence:** 0.872

### Performance Metrics
| Metric | Value |
| --- | --- |
| Retrieval Time | 123.4 ms |
| Generation Time | 2087.3 ms |
| Total Time | 2210.7 ms |

### Quality Metrics
| Metric | Score |
| --- | --- |
| Accuracy | 0.892 |
| F1 Score | 0.821 |
| BLEU | 0.456 |
| GLEU | 0.534 |
| ROUGE-1 | 0.823 |
| ROUGE-L | 0.745 |
| BERTScore | 0.834 |
| S-BERT Similarity | 0.876 |
| DISTINCT | 0.854 |
| LLM Judge (Norm) | 0.854 |

### RAGAS Metrics (RAG-Specific)
| Metric | Score |
| --- | --- |
| Faithfulness | 0.823 |
| Answer Relevance | 0.879 |
| Context Relevance | 0.801 |

### Quick Assessment
✅ **Good overall quality** - Strong text similarity metrics  
✅ **Well-evaluated by LLM** - Human-level quality

---

## Experiment 2 (No LAQA)

### Sample Statistics
- **Samples Evaluated:** 10
- **Blocked Rate:** 0.100 (10%)
- **Average Confidence:** 0.865

### Performance Metrics
| Metric | Value |
| --- | --- |
| Retrieval Time | 41.2 ms |
| Generation Time | 2089.1 ms |
| Total Time | 2130.3 ms |

### Quality Metrics
| Metric | Score |
| --- | --- |
| Accuracy | 0.876 |
| F1 Score | 0.789 |
| BLEU | 0.421 |
| GLEU | 0.498 |
| ROUGE-1 | 0.798 |
| ROUGE-L | 0.712 |
| BERTScore | 0.805 |
| S-BERT Similarity | 0.843 |
| DISTINCT | 0.831 |
| LLM Judge (Norm) | 0.821 |

### RAGAS Metrics (RAG-Specific)
| Metric | Score |
| --- | --- |
| Faithfulness | 0.791 |
| Answer Relevance | 0.856 |
| Context Relevance | 0.768 |

### Quick Assessment
✅ **Good overall quality** - Strong text similarity metrics  
✅ **Well-evaluated by LLM** - Human-level quality

---

## Experiment 3 (No LAQA + No MRL)

### Sample Statistics
- **Samples Evaluated:** 10
- **Blocked Rate:** 0.100 (10%)
- **Average Confidence:** 0.858

### Performance Metrics
| Metric | Value |
| --- | --- |
| Retrieval Time | 89.7 ms |
| Generation Time | 2091.5 ms |
| Total Time | 2181.2 ms |

### Quality Metrics
| Metric | Score |
| --- | --- |
| Accuracy | 0.864 |
| F1 Score | 0.776 |
| BLEU | 0.408 |
| GLEU | 0.481 |
| ROUGE-1 | 0.782 |
| ROUGE-L | 0.698 |
| BERTScore | 0.791 |
| S-BERT Similarity | 0.829 |
| DISTINCT | 0.812 |
| LLM Judge (Norm) | 0.809 |

### RAGAS Metrics (RAG-Specific)
| Metric | Score |
| --- | --- |
| Faithfulness | 0.775 |
| Answer Relevance | 0.843 |
| Context Relevance | 0.754 |

### Quick Assessment
✅ **Good overall quality** - Strong text similarity metrics  
✅ **Well-evaluated by LLM** - Human-level quality

---

## Experiment 4 (No RAG - Baseline)

### Sample Statistics
- **Samples Evaluated:** 10
- **Blocked Rate:** 0.100 (10%)
- **Average Confidence:** 0.743

### Performance Metrics
| Metric | Value |
| --- | --- |
| Retrieval Time | 0.0 ms |
| Generation Time | 2084.6 ms |
| Total Time | 2084.6 ms |

### Quality Metrics
| Metric | Score |
| --- | --- |
| Accuracy | 0.712 |
| F1 Score | 0.654 |
| BLEU | 0.289 |
| GLEU | 0.345 |
| ROUGE-1 | 0.634 |
| ROUGE-L | 0.523 |
| BERTScore | 0.689 |
| S-BERT Similarity | 0.701 |
| DISTINCT | 0.721 |
| LLM Judge (Norm) | 0.675 |

### RAGAS Metrics (RAG-Specific)
| Metric | Score |
| --- | --- |
| Faithfulness | 0.521 |
| Answer Relevance | 0.712 |
| Context Relevance | 0.134 |

### Quick Assessment
⚠️ **Moderate quality** - Some text divergence from reference  
⚠️ **Average LLM evaluation** - Room for improvement

---

## Key Insights

**Best F1 Score:** Experiment 1 (0.821)  
**Best Latency:** Experiment 4 (2084.6 ms)  
**Best LLM Judge:** Experiment 1 (0.854)

### Quality vs Performance Trade-offs

**RAG Contribution (Exp1 vs Exp4):**
- F1 improvement: 25.5%
- Latency cost: +126.1 ms

**LAQA Contribution (Exp1 vs Exp2):**
- F1 improvement: 4.1%
- Latency cost: +80.4 ms

### Recommendations

- Choose experiment with best **LLM Judge score** for highest human-aligned quality
- Consider **latency constraints** when selecting between high-quality and fast options
- Monitor **faithfulness score** to ensure answers are grounded in retrieved context
- Check **blocked rate** if safety is critical for your use case

---

## Analysis Summary

### Quality Hierarchy
1. 🥇 **Experiment 1** (Full Pipeline): Best quality, balanced metrics
2. 🥈 **Experiment 2** (No LAQA): 99% of quality at 96% latency cost
3. 🥉 **Experiment 3** (No MRL): Comparable to Exp2, slightly slower
4. ❌ **Experiment 4** (No RAG): Baseline - significantly lower quality

### Recommendation by Use Case

**Medical Quality Critical** → Use **Experiment 1**
- Highest faithfulness (0.823)
- Best LLM evaluation (0.854)
- Acceptable latency (~2.2s)

**Speed Important** → Use **Experiment 2**
- Only 4% quality loss vs Exp1
- 38% faster (80ms saved)
- Still strong metrics (F1: 0.789)

**Cost Conscious** → Use **Experiment 3**
- Similar speed to Exp2
- Full embeddings vs truncated
- Minimal quality difference

**NOT RECOMMENDED** → **Experiment 4**
- 25% quality loss vs Exp1
- No latency benefit (LLM still takes ~2s)
- Significantly lower faithfulness

---

*Generated from ablation study with 10 medical questions*  
*Each experiment evaluated on accuracy, semantic similarity, and LLM-based metrics*
