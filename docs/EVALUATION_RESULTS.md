# Evaluation Results

## RAGAS Metrics

Evaluation framework: [RAGAS](https://docs.ragas.io/) v0.4

### Target Metrics

| Metric             | Target | Description                        |
|--------------------|--------|------------------------------------|
| Faithfulness       | > 0.80 | Is the answer grounded in sources? |
| Answer Relevancy   | > 0.80 | Does it answer the question?       |
| Context Precision  | > 0.80 | Are retrieved chunks relevant?     |
| Context Recall     | > 0.80 | Were all relevant chunks found?    |

### Custom Metrics

| Metric    | Description                                  |
|-----------|----------------------------------------------|
| Hit Rate  | % of queries where correct context retrieved |
| MRR       | Mean Reciprocal Rank of first relevant chunk |

## Running Evaluation

```bash
# Generate synthetic test data from a contract
poetry run python -c "
from src.evaluation.synthetic_data import generate_synthetic_testset
from src.ingestion.loaders import load_document

doc = load_document('data/sample_contracts/test_nda.pdf')
generate_synthetic_testset(doc.full_text, n=10, output_path='data/synthetic_eval/testset.json')
"

# Run RAGAS evaluation
poetry run python -c "
from src.evaluation.evaluator import evaluate_from_testset
results = evaluate_from_testset('data/synthetic_eval/testset.json', output_path='data/synthetic_eval/results.json')
print(results)
"
```

## Methodology

1. Synthetic test data generated using GPT-4o-mini from sample contracts
2. Each sample has: question, expected answer, ground truth context
3. RAG pipeline processes each question to get retrieved contexts and generated answers
4. RAGAS metrics computed using LLM-as-judge (GPT-4o-mini)
5. Custom metrics (hit rate, MRR) computed with string matching

## Notes

- Start with 10 test cases to minimize API costs
- Results may vary based on contract complexity
- Iterate on prompts and retrieval parameters to improve metrics
