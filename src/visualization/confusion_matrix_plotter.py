"""Confusion matrix heatmaps for experiment history."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_confusion_matrices_by_category(
    categories: Optional[Iterable[str]] = None,
    model_hf_id: Optional[str] = None,
    prompt_template: Optional[str] = None,
    output_format: Optional[str] = None,
    experiments_dir: str = "results/experiment_history",
    max_samples: Optional[int] = None,
    quantization: str = "8bit",
    text_max_tokens: int = 5,
    json_max_tokens: int = 15,
    selection_metric: str = "qwk",
    use_standard_config: bool = True,
    figsize_per_plot: tuple = (4.5, 4.0),
) -> Optional[pd.DataFrame]:
    """Plot confusion matrix heatmaps for selected categories.

    By default, this mirrors the standard configuration filters used in
    summary charts: full test set, 8-bit quantization, and token limits.
    """
    exp_base = Path(experiments_dir)
    if not exp_base.exists():
        print("No experiment history found. Run Phase 1, 2, or 3 first.")
        return None

    category_files = list(exp_base.glob("*.json"))
    if not category_files:
        print("No experiment data found.")
        return None

    category_set = set(categories) if categories else None
    all_rows = []

    for category_file in category_files:
        category = category_file.stem
        if category_set is not None and category not in category_set:
            continue

        try:
            with open(category_file, "r", encoding="utf-8") as f:
                experiments = json.load(f)
        except Exception:
            continue

        for exp in experiments:
            metrics = exp.get("metrics", {})
            confusion_matrix = metrics.get("confusion_matrix")
            if confusion_matrix is None:
                continue

            model = exp.get("model", "unknown")
            if model_hf_id and model != model_hf_id:
                continue

            exp_template = exp.get("prompt_template", "unknown")
            if prompt_template and exp_template != prompt_template:
                continue

            exp_output_format = exp.get("output_format", "text")
            if output_format and exp_output_format != output_format:
                continue

            dataset_info = exp.get("dataset", {})
            exp_max_samples = dataset_info.get("max_samples")
            exp_quantization = exp.get("quantization", "none")
            generation_params = exp.get("generation_params", {})
            max_new_tokens = generation_params.get("max_new_tokens")

            if max_samples is not None and exp_max_samples != max_samples:
                continue

            if use_standard_config:
                if exp_max_samples is not None:
                    continue
                if exp_quantization != quantization:
                    continue
                if exp_output_format == "text" and max_new_tokens != text_max_tokens:
                    continue
                if exp_output_format == "json" and max_new_tokens != json_max_tokens:
                    continue

            row = {
                "category": category,
                "template": exp_template,
                "output_format": exp_output_format,
                "model": model,
                "quantization": exp_quantization,
                "selection_metric": metrics.get(selection_metric),
                "confusion_matrix": confusion_matrix,
            }
            all_rows.append(row)

    if not all_rows:
        print("No matching experiments with confusion matrices found.")
        return None

    df = pd.DataFrame(all_rows)
    df["selection_metric"] = pd.to_numeric(df["selection_metric"], errors="coerce")
    df = df.dropna(subset=["selection_metric"])

    if df.empty:
        print("No experiments with a valid selection metric found.")
        return None

    best_by_category = df.loc[df.groupby("category")["selection_metric"].idxmax()].copy()
    best_by_category = best_by_category.sort_values("category")

    _plot_confusion_matrices(
        best_by_category,
        figsize_per_plot,
        selection_metric,
        model_hf_id=model_hf_id,
        prompt_template=prompt_template,
        output_format=output_format,
        use_standard_config=use_standard_config,
    )
    return best_by_category.reset_index(drop=True)


def _plot_confusion_matrices(
    best_by_category: pd.DataFrame,
    figsize_per_plot: tuple,
    selection_metric: str,
    model_hf_id: Optional[str],
    prompt_template: Optional[str],
    output_format: Optional[str],
    use_standard_config: bool,
) -> None:
    categories = best_by_category["category"].tolist()
    n_plots = len(categories)

    ncols = min(3, n_plots)
    nrows = int(np.ceil(n_plots / ncols))

    fig_width = figsize_per_plot[0] * ncols
    fig_height = figsize_per_plot[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    axes = np.array(axes).reshape(-1)

    model_label = model_hf_id or "All models"
    template_label = prompt_template or "Any template"
    output_label = output_format or "Any format"
    config_label = "standard config" if use_standard_config else "all configs"
    fig.suptitle(
        "Confusion Matrix Heatmaps (best run per category)",
        fontsize=14,
        fontweight="bold",
        y=0.968,
    )
    fig.text(
        0.5,
        0.952,
        f"Model: {model_label} | Template: {template_label} | "
        f"Output: {output_label} | Config: {config_label} | "
        f"Selected by: {selection_metric.upper()}",
        ha="center",
        va="top",
        fontsize=10,
    )

    for idx, (_, row) in enumerate(best_by_category.iterrows()):
        ax = axes[idx]
        matrix = np.array(row["confusion_matrix"], dtype=float)
        im = ax.imshow(matrix, cmap="Blues")

        ax.set_title(
            f"{row['category']} | {row['template']} | {row['output_format']}",
            fontsize=9,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        ticks = np.arange(matrix.shape[0])
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = int(matrix[i, j]) if matrix[i, j].is_integer() else matrix[i, j]
                ax.text(j, i, value, ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[n_plots:]:
        ax.axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.925))
    plt.subplots_adjust(top=0.925)
    plt.show()
