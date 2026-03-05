"""RAGAS evaluation runner."""

import json
from pathlib import Path

from loguru import logger
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from config.settings import settings
from src.evaluation.metrics import _is_match, mean_reciprocal_rank, retrieval_hit_rate


def load_testset(path: str | Path) -> list[dict]:
    """Load a testset from JSON file."""
    data = json.loads(Path(path).read_text())
    return data.get("samples", [])


def run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    ground_truth_contexts: list[str] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, float]:
    """Run RAGAS evaluation on provided data.

    Args:
        questions: List of questions.
        answers: List of generated answers.
        contexts: List of retrieved context lists (per question).
        ground_truths: List of ground truth answers.
        ground_truth_contexts: List of ground truth context strings
            (used for hit_rate/mrr). Falls back to ground_truths if not provided.
        output_path: Optional path to save results CSV.

    Returns:
        Dict of metric names to scores.
    """
    logger.info(f"Running RAGAS evaluation on {len(questions)} samples...")

    samples = []
    for q, a, c, gt in zip(questions, answers, contexts, ground_truths):
        samples.append(
            SingleTurnSample(
                user_input=q,
                response=a,
                retrieved_contexts=c,
                reference=gt,
            )
        )

    dataset = EvaluationDataset(samples=samples)

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    judge_model = settings.RAGAS_JUDGE_MODEL
    if judge_model == settings.OPENAI_CHAT_MODEL:
        logger.warning(
            f"RAGAS judge model ({judge_model}) is the same as the generator model. "
            "For production evaluation, set RAGAS_JUDGE_MODEL to a stronger model "
            "(e.g. gpt-4o) for more reliable metrics."
        )

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=judge_model,
            api_key=settings.OPENAI_API_KEY,
            max_retries=10,
            request_timeout=120,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
            max_retries=10,
            request_timeout=60,
        )
    )

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    logger.info(
        f"Starting RAGAS with judge={judge_model}, "
        f"max_retries=10, exponential backoff enabled"
    )
    result = evaluate(dataset=dataset, metrics=metrics, batch_size=2)

    # Extract scores from EvaluationResult
    scores = {}
    import math

    per_sample_scores: list[dict] | None = None

    if isinstance(result, dict):
        result_dict = result
    elif hasattr(result, "scores"):
        # RAGAS >= 0.2: .scores can be list[dict] or dict
        raw = result.scores
        if isinstance(raw, list):
            # List of per-sample dicts — aggregate means (skip NaN values)
            from collections import defaultdict
            agg = defaultdict(list)
            nan_counts: dict[str, int] = defaultdict(int)
            per_sample_scores = []
            for i, row in enumerate(raw):
                if not isinstance(row, dict):
                    continue
                per_sample_scores.append(row)
                for k, v in row.items():
                    if isinstance(v, (int, float)):
                        if math.isnan(v):
                            nan_counts[k] += 1
                            logger.warning(
                                f"Sample {i+1}: {k} = NaN "
                                f"(question: '{questions[i][:50]}...')"
                            )
                        else:
                            agg[k].append(v)
            if nan_counts:
                logger.warning(
                    f"NaN summary: {dict(nan_counts)} — "
                    f"these samples were excluded from averages"
                )
            result_dict = {k: sum(v) / len(v) for k, v in agg.items() if v}
        else:
            result_dict = raw
    elif hasattr(result, "__iter__"):
        result_dict = dict(result)
    else:
        logger.warning(f"Unknown EvaluationResult type: {type(result)}")
        result_dict = {}

    for key, value in result_dict.items():
        if isinstance(value, (int, float)):
            scores[key] = round(float(value), 4)

    # Add custom metrics (compare retrieved contexts against ground truth contexts)
    gt_ctx = ground_truth_contexts if ground_truth_contexts is not None else ground_truths
    scores["hit_rate"] = round(retrieval_hit_rate(contexts, gt_ctx), 4)
    scores["mrr"] = round(mean_reciprocal_rank(contexts, gt_ctx), 4)

    logger.info(f"RAGAS results: {scores}")

    # Per-sample diagnostics dump for deeper analysis
    try:
        diagnostics_dir = settings.DATA_DIR / "synthetic_eval"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        per_sample_path = diagnostics_dir / "per_sample.jsonl"

        with per_sample_path.open("w", encoding="utf-8") as f:
            for idx, (q, a, ctx_list, gt_ans, gt_c) in enumerate(
                zip(questions, answers, contexts, ground_truths, gt_ctx)
            ):
                first_match_rank: int | None = None
                has_match = False
                for rank, chunk in enumerate(ctx_list, 1):
                    if _is_match(chunk, gt_c or gt_ans, threshold=0.3):
                        first_match_rank = rank
                        has_match = True
                        break

                row: dict = {
                    "index": idx,
                    "question": q,
                    "answer": a,
                    "ground_truth_answer": gt_ans,
                    "ground_truth_context": gt_c,
                    "retrieved_contexts": ctx_list,
                    "first_match_rank": first_match_rank,
                    "has_match": has_match,
                }

                if per_sample_scores and idx < len(per_sample_scores):
                    row["ragas"] = per_sample_scores[idx]

                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        logger.info(f"Per-sample diagnostics written to {per_sample_path}")
    except Exception as e:  # pragma: no cover - diagnostics should not break eval
        logger.warning(f"Failed to write per-sample diagnostics: {e}")

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scores, indent=2))
        logger.info(f"Results saved to {path}")

    return scores


def evaluate_from_testset(
    testset_path: str | Path,
    retriever_fn=None,
    qa_fn=None,
    output_path: str | Path | None = None,
) -> dict[str, float]:
    """Run full evaluation from a testset file.

    If retriever_fn and qa_fn are provided, runs the full pipeline.
    Otherwise expects pre-computed answers in the testset.
    """
    samples = load_testset(testset_path)
    if not samples:
        logger.error("Empty testset")
        return {}

    questions = [s["question"] for s in samples]
    ground_truths = [s["answer"] for s in samples]
    gt_contexts = [s.get("context", "") for s in samples]

    if retriever_fn and qa_fn:
        answers = []
        contexts = []
        for q in questions:
            ctx = retriever_fn(q)
            contexts.append(ctx)
            ans = qa_fn(q, "\n".join(ctx))
            answers.append(ans)
    else:
        # Use ground truth as both answer and context for baseline
        answers = ground_truths
        contexts = [[c] for c in gt_contexts]

    return run_ragas_evaluation(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
        ground_truth_contexts=gt_contexts,
        output_path=output_path,
    )
