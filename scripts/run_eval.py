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


def run_pipeline(
    testset_path: str,
    output_path: str,
    contexts_per_query: int | None = None,
    eval_retrieval_top_k: int | None = None,
    eval_reranker_threshold: float | None = None,
) -> dict[str, float]:
    """Run full pipeline evaluation with retriever + QA agent.

    This function can optionally override retrieval parameters just for the
    evaluation run (e.g., lower RETRIEVAL_TOP_K or increase the reranker
    score threshold) and limit the number of contexts passed per query.
    """
    from src.agents.qa_agent import QAAgent
    from src.evaluation.evaluator import evaluate_from_testset
    from src.retrieval.retriever import Retriever
    from config.settings import settings

    logger.info("Running PIPELINE evaluation (retriever + QA agent)...")

    # Optionally override retrieval params only for this evaluation run
    original_top_k = settings.RETRIEVAL_TOP_K
    original_threshold = getattr(settings, "RERANKER_SCORE_THRESHOLD", None)
    if eval_retrieval_top_k is not None:
        logger.info(
            f"Overriding RETRIEVAL_TOP_K for evaluation: "
            f"{original_top_k} -> {eval_retrieval_top_k}"
        )
        settings.RETRIEVAL_TOP_K = eval_retrieval_top_k
    if eval_reranker_threshold is not None and original_threshold is not None:
        logger.info(
            f"Overriding RERANKER_SCORE_THRESHOLD for evaluation: "
            f"{original_threshold} -> {eval_reranker_threshold}"
        )
        settings.RERANKER_SCORE_THRESHOLD = eval_reranker_threshold

    try:
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

            if contexts_per_query is not None and contexts_per_query > 0:
                return texts[:contexts_per_query]
            return texts

        def qa_fn(question: str, context: str) -> str:
            return qa_agent.answer(question, context).answer

        return evaluate_from_testset(
            testset_path=testset_path,
            retriever_fn=retriever_fn,
            qa_fn=qa_fn,
            output_path=output_path,
        )
    finally:
        # Restore original settings after evaluation
        settings.RETRIEVAL_TOP_K = original_top_k
        if original_threshold is not None:
            settings.RERANKER_SCORE_THRESHOLD = original_threshold


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
    parser.add_argument(
        "--contexts-per-query",
        type=int,
        default=None,
        help="Limit number of contexts per question passed to RAGAS (e.g. 1 for top-1 only).",
    )
    parser.add_argument(
        "--eval-retrieval-top-k",
        type=int,
        default=None,
        help="Override RETRIEVAL_TOP_K only for this evaluation run.",
    )
    parser.add_argument(
        "--eval-reranker-threshold",
        type=float,
        default=None,
        help="Override RERANKER_SCORE_THRESHOLD only for this evaluation run.",
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
        results = run_pipeline(
            testset_path=testset_path,
            output_path=output_path,
            contexts_per_query=args.contexts_per_query,
            eval_retrieval_top_k=args.eval_retrieval_top_k,
            eval_reranker_threshold=args.eval_reranker_threshold,
        )

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
