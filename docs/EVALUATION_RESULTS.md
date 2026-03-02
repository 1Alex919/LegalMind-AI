# Evaluation Results

## RAGAS Metrics

Evaluation framework: [RAGAS](https://docs.ragas.io/) v0.4

### Current Results (bundled NDA synthetic testset)

Results for `data/synthetic_eval/testset.json` using `scripts/run_eval.py --mode pipeline`:

| Metric             | Score  | Target | Description                        |
|--------------------|--------|--------|------------------------------------|
| Faithfulness       | 0.9333 | > 0.80 | Is the answer grounded in sources? |
| Answer Relevancy   | 0.7306 | > 0.80 | Does it answer the question?       |
| Context Precision  | 0.3148 | > 0.80 | Are retrieved chunks relevant?     |
| Context Recall     | 0.6111 | > 0.80 | Were all relevant chunks found?    |
| Hit Rate           | 0.6667 | > 0.80 | Does correct context appear?       |
| MRR                | 0.3796 | > 0.70 | Rank of first relevant result      |

### Custom Metrics

Custom retrieval metrics are computed in `src/evaluation/metrics.py` and use fuzzy matching:

| Metric    | Description                                                  |
|-----------|--------------------------------------------------------------|
| Hit Rate  | % of queries where at least one retrieved chunk matches truth|
| MRR       | Mean reciprocal rank of the first chunk that matches truth   |

## Running Evaluation

```bash
# Generate synthetic test data from a contract
python scripts/generate_testset.py --file data/sample_contracts/contract.pdf \
  --samples 20 \
  --output data/synthetic_eval/testset.json

# Run baseline evaluation (ground truth as context/answers)
python scripts/run_eval.py \
  --mode baseline \
  --testset data/synthetic_eval/testset.json \
  --output data/synthetic_eval/results_baseline.json

# Run full RAG pipeline evaluation (retriever + QA agent)
python scripts/run_eval.py \
  --mode pipeline \
  --testset data/synthetic_eval/testset.json \
  --output data/synthetic_eval/results.json
```

## Methodology

1. Synthetic test data generated using GPT-4o from sample contracts (via `scripts/generate_testset.py`).
2. Each sample has: question, expected answer, ground truth context.
3. Full RAG pipeline (hybrid search + query expansion + reranking + QA agent) processes each question to get retrieved contexts and generated answers.
4. RAGAS metrics computed using LLM-as-judge (GPT-4o) with `max_retries=10`, `batch_size=2`, and exponential backoff.
5. Custom metrics (hit rate, MRR) computed with fuzzy token matching (substring, token containment, Jaccard similarity).

## Notes

- Start with a small number of test cases to minimize API costs.
- Results may vary based on contract complexity and dataset composition.
- Iterate on prompts, retrieval parameters (BM25, hybrid alpha, reranker thresholds), and chunking strategy to improve metrics.
