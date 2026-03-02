"""Run RAGAS evaluation on the synthetic testset.

Usage:
    # Baseline (ground truth as context/answers — no retrieval needed):
    python scripts/run_eval.py

    # Full pipeline (requires ingested documents in ChromaDB):
    python scripts/run_eval.py --mode pipeline

    # Custom testset path:
    python scripts/run_eval.py --testset data/synthetic_eval/testset.json

    # Custom output path:
    python scripts/run_eval.py --output data/synthetic_eval/results.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger


def run_baseline(testset_path: str, output_path: str) -> dict[str, float]:
    """Run baseline evaluation using ground truth data (no retrieval)."""
    from src.evaluation.evaluator import evaluate_from_testset

    logger.info("Running BASELINE evaluation (ground truth as context)...")
    return evaluate_from_testset(
        testset_path=testset_path,
        output_path=output_path,
    )


def run_pipeline(testset_path: str, output_path: str) -> dict[str, float]:
    """Run full pipeline evaluation with retriever + QA agent."""
    from src.agents.qa_agent import QAAgent
    from src.evaluation.evaluator import evaluate_from_testset
    from src.retrieval.retriever import Retriever

    logger.info("Running PIPELINE evaluation (retriever + QA agent)...")

    retriever = Retriever()
    qa_agent = QAAgent()

    def retriever_fn(query: str) -> list[str]:
        result = retriever.retrieve(query)
        texts: list[str] = []
        seen_parents: set[str] = set()

        for r in result.results:
            pid = r.metadata.get("parent_id")
            if pid and pid in result.parent_context and pid not in seen_parents:
                texts.append(result.parent_context[pid])
                seen_parents.add(pid)
            else:
                texts.append(r.text)

        return texts

    def qa_fn(question: str, context: str) -> str:
        return qa_agent.answer(question, context).answer

    return evaluate_from_testset(
        testset_path=testset_path,
        retriever_fn=retriever_fn,
        qa_fn=qa_fn,
        output_path=output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument(
        "--mode",
        choices=["baseline", "pipeline"],
        default="baseline",
        help="Evaluation mode: 'baseline' uses ground truth, 'pipeline' uses full RAG (default: baseline)",
    )
    parser.add_argument(
        "--testset",
        default="data/synthetic_eval/testset.json",
        help="Path to testset JSON (default: data/synthetic_eval/testset.json)",
    )
    parser.add_argument(
        "--output",
        default="data/synthetic_eval/results.json",
        help="Path to save results (default: data/synthetic_eval/results.json)",
    )
    args = parser.parse_args()

    testset_path = str(PROJECT_ROOT / args.testset)
    output_path = str(PROJECT_ROOT / args.output)

    if not Path(testset_path).exists():
        logger.error(f"Testset not found: {testset_path}")
        logger.info("Generate one first: python scripts/generate_testset.py --file <contract.pdf>")
        sys.exit(1)

    if args.mode == "baseline":
        results = run_baseline(testset_path, output_path)
    else:
        results = run_pipeline(testset_path, output_path)

    # Print results table
    print("\n" + "=" * 50)
    print("  RAGAS Evaluation Results")
    print("=" * 50)
    for metric, score in results.items():
        bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
        status = "✓" if score >= 0.8 else "△" if score >= 0.5 else "✗"
        print(f"  {status} {metric:<22} {bar} {score:.4f}")
    print("=" * 50)
    print(f"\n  Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
