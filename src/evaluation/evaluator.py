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
from src.evaluation.metrics import mean_reciprocal_rank, retrieval_hit_rate


def load_testset(path: str | Path) -> list[dict]:
    """Load a testset from JSON file."""
    data = json.loads(Path(path).read_text())
    return data.get("samples", [])


def run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    output_path: str | Path | None = None,
) -> dict[str, float]:
    """Run RAGAS evaluation on provided data.

    Args:
        questions: List of questions.
        answers: List of generated answers.
        contexts: List of retrieved context lists (per question).
        ground_truths: List of ground truth answers.
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
        ChatOpenAI(model=judge_model, api_key=settings.OPENAI_API_KEY)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )
    )

    metrics = [
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
    ]

    result = evaluate(dataset=dataset, metrics=metrics)

    # Extract scores from EvaluationResult
    scores = {}
    if isinstance(result, dict):
        result_dict = result
    elif hasattr(result, "scores"):
        # RAGAS >= 0.2: .scores can be list[dict] or dict
        raw = result.scores
        if isinstance(raw, list):
            # List of per-sample dicts — aggregate means
            from collections import defaultdict
            agg = defaultdict(list)
            for row in raw:
                if isinstance(row, dict):
                    for k, v in row.items():
                        if isinstance(v, (int, float)):
                            agg[k].append(v)
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

    # Add custom metrics
    scores["hit_rate"] = round(retrieval_hit_rate(contexts, ground_truths), 4)
    scores["mrr"] = round(mean_reciprocal_rank(contexts, ground_truths), 4)

    logger.info(f"RAGAS results: {scores}")

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
        output_path=output_path,
    )
