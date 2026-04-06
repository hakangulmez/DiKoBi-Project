"""Classification module for DiKoBi text classification."""

from .prompts import CATEGORIES, get_prompt, get_valid_outputs
from .classifier import TextClassifier


def list_categories():
    """Return list of available category names."""
    return list(CATEGORIES.keys())


__all__ = ["CATEGORIES", "get_prompt", "get_valid_outputs", "TextClassifier", "list_categories"]
