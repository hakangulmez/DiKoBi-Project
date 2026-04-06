"""Evaluation tools for DiKoBi classification system."""

from .metrics import calculate_metrics, print_metrics_summary
from .compare_templates import PromptTester, compare_all_templates, test_custom_prompt
from .compare_models import ModelComparison, ModelComparisonConfig, compare_all_models
from .rag_evaluator import (
    evaluate_rag_single_category,
    print_results_summary,
    build_predictions_list,
)
from .batch_rag_evaluator import (
    evaluate_rag_all_categories,
    evaluate_rag_categories_list,
)

__all__ = [
    "calculate_metrics",
    "print_metrics_summary",
    "PromptTester",
    "compare_all_templates",
    "test_custom_prompt",
    "ModelComparison",
    "ModelComparisonConfig",
    "compare_all_models",
    "evaluate_rag_single_category",
    "print_results_summary",
    "build_predictions_list",
    "evaluate_rag_all_categories",
    "evaluate_rag_categories_list",
]
