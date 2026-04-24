"""DeepEval test cases for RAG pipeline evaluation.

Measures hallucination, faithfulness, and contextual relevancy.
Run with: deepeval test run tests/test_rag.py
"""

import json
import csv
import os
from datetime import datetime

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    HallucinationMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)

from app.rag_pipeline import query, get_retriever


# --- Load test dataset ---
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_dataset.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "evaluation", "eval_results.csv")


def load_test_dataset():
    """Load ground-truth Q&A pairs from JSON file."""
    with open(TEST_DATA_PATH, "r") as f:
        return json.load(f)


def get_context_for_question(question: str) -> list[str]:
    """Retrieve context chunks for a question."""
    retriever = get_retriever()
    docs = retriever.invoke(question)
    return [doc.page_content for doc in docs]


def log_result(question: str, scores: dict):
    """Append evaluation scores to CSV for tracking over time."""
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    file_exists = os.path.exists(RESULTS_PATH)

    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "question",
                "hallucination_score", "faithfulness_score", "relevancy_score",
            ])
        writer.writerow([
            datetime.now().isoformat(),
            question,
            scores.get("hallucination", "N/A"),
            scores.get("faithfulness", "N/A"),
            scores.get("relevancy", "N/A"),
        ])


# --- Test cases ---

dataset = load_test_dataset()


@pytest.mark.parametrize("test_case", dataset, ids=[tc["id"] for tc in dataset])
def test_rag_hallucination(test_case):
    """Test that the RAG pipeline does not hallucinate."""
    # Get the actual output from the pipeline
    result = query(test_case["question"])
    actual_output = result["answer"]

    # Get retrieved context
    context = get_context_for_question(test_case["question"])

    # Build DeepEval test case
    eval_case = LLMTestCase(
        input=test_case["question"],
        actual_output=actual_output,
        expected_output=test_case["expected_answer"],
        context=context,
    )

    # Run metrics
    hallucination_metric = HallucinationMetric(threshold=0.5)
    faithfulness_metric = FaithfulnessMetric(threshold=0.5)
    relevancy_metric = ContextualRelevancyMetric(threshold=0.5)

    scores = {}
    try:
        hallucination_metric.measure(eval_case)
        scores["hallucination"] = hallucination_metric.score
    except Exception:
        scores["hallucination"] = "error"

    try:
        faithfulness_metric.measure(eval_case)
        scores["faithfulness"] = faithfulness_metric.score
    except Exception:
        scores["faithfulness"] = "error"

    try:
        relevancy_metric.measure(eval_case)
        scores["relevancy"] = relevancy_metric.score
    except Exception:
        scores["relevancy"] = "error"

    # Log results
    log_result(test_case["question"], scores)

    # Assert using DeepEval
    assert_test(eval_case, [hallucination_metric])
