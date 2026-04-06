"""Plotting functions for experiment results visualization."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional
import json

# Define custom order once - used throughout the module
COMBO_ORDER = [
    ('zero_shot', 'text'),
    ('few_shot', 'text'),
    ('zero_shot', 'json'),
    ('few_shot', 'json'),
    ('rag_balanced', 'text'),
    ('rag_similarity', 'text'),
    ('rag_balanced', 'json'),
    ('rag_similarity', 'json'),
]

# Color scheme for each combination
COLOR_MAP = {
    ('zero_shot', 'text'): "#186D92",      # Blue
    ('few_shot', 'text'): '#C71717',       # Red
    ('zero_shot', 'json'): '#F18F01',      # Orange
    ('few_shot', 'json'): '#068148',       # Green
    ('rag_balanced', 'text'): '#9B59B6',   # Purple
    ('rag_similarity', 'text'): "#FFF01B",
    ('rag_balanced', 'json'): "#06A1A7",   # Light Blue
    ('rag_similarity', 'json'): "#FA76C8", # Coral Red
}

# Display labels for templates
TEMPLATE_LABELS = {
    ('zero_shot', 'text'): 'Zero-Shot (TEXT)',
    ('few_shot', 'text'): 'Few-Shot (TEXT)',
    ('zero_shot', 'json'): 'Zero-Shot (JSON)',
    ('few_shot', 'json'): 'Few-Shot (JSON)',
    ('rag_balanced', 'text'): 'RAG Balanced (TEXT)',
    ('rag_similarity', 'text'): 'RAG Similarity (TEXT)',
    ('rag_balanced', 'json'): 'RAG Balanced (JSON)',
    ('rag_similarity', 'json'): 'RAG Similarity (JSON)',
}


def plot_best_qwk_by_category(
    model_hf_id: Optional[str] = None,
    experiments_dir: str = "results/experiment_history",
    figsize: tuple = (24, 10)  # Increased width from 20 to 24, height from 8 to 10
) -> Optional[pd.DataFrame]:
    """
    Visualize best QWK scores across categories, templates, and output formats.
    
    Args:
        model_hf_id: HuggingFace model ID to filter results (e.g., "Qwen/Qwen2.5-7B-Instruct").
                     If None, shows results for all models.
        experiments_dir: Path to experiment history directory.
        figsize: Figure size for the plot.
    """
    best_results = _collect_best_results(model_hf_id, experiments_dir, announce=True)
    if best_results is None:
        return
    
    # Create visualization
    _create_bar_chart(best_results, model_hf_id, figsize)
    
    # Print summary statistics
    summary = _print_summary_statistics(best_results, model_hf_id)
    _plot_summary_panels(summary, model_hf_id, figsize=(18, 10))
    return summary


def plot_summary_statistics(
    model_hf_id: Optional[str] = None,
    experiments_dir: str = "results/experiment_history",
    figsize: tuple = (18, 10)
) -> Optional[pd.DataFrame]:
    """Plot weighted summary statistics across templates and output formats."""
    best_results = _collect_best_results(model_hf_id, experiments_dir, announce=True)
    if best_results is None:
        return

    summary = _print_summary_statistics(best_results, model_hf_id)
    _plot_summary_panels(summary, model_hf_id, figsize)
    return summary


def _collect_best_results(
    model_hf_id: Optional[str],
    experiments_dir: str,
    announce: bool
) -> Optional[pd.DataFrame]:
    if announce:
        print("📊 Analyzing experiments across categories...\n")

    exp_base = Path(experiments_dir)
    if not exp_base.exists():
        print("❌ No experiment history found. Run Phase 1, 2, or 3 first.")
        return None

    category_files = list(exp_base.glob("*.json"))
    if not category_files:
        print("❌ No experiment data found.")
        return None

    all_results = []
    for category_file in category_files:
        category = category_file.stem
        
        try:
            with open(category_file, 'r', encoding='utf-8') as f:
                experiments = json.load(f)
            
            # Each file contains a list of experiments for that category
            for exp in experiments:
                # Access metrics correctly
                metrics = exp.get('metrics', {})
                qwk = metrics.get('qwk')
                accuracy = metrics.get('accuracy')
                mse = metrics.get('mse')
                mae = metrics.get('mae')
                
                # Skip if no QWK score
                if qwk is None:
                    continue
                
                model = exp.get('model', 'unknown')
                
                # Filter by model if specified
                if model_hf_id and model != model_hf_id:
                    continue
                
                # Filter for standard configuration only:
                # - Full test set (max_samples = None)
                # - 8-bit quantization enabled
                # - Text max tokens = 5, JSON max tokens = 15
                dataset_info = exp.get('dataset', {})
                max_samples = dataset_info.get('max_samples')
                samples_used = dataset_info.get('samples_used')
                total_samples = dataset_info.get('total_samples')
                quantization = exp.get('quantization', 'none')
                generation_params = exp.get('generation_params', {})
                max_new_tokens = generation_params.get('max_new_tokens')
                output_format = exp.get('output_format', 'text')
                
                # Skip experiments that don't match the standard configuration
                if max_samples is not None:  # Must be full test set (None)
                    continue
                if quantization != '8bit':  # Must use 8-bit quantization
                    continue
                if output_format == 'text' and max_new_tokens != 5:  # Text format must use 5 tokens
                    continue
                if output_format == 'json' and max_new_tokens != 15:  # JSON format must use 15 tokens
                    continue
                
                all_results.append({
                    'category': category,
                    'template': exp.get('prompt_template', 'unknown'),
                    'output_format': output_format,
                    'qwk': qwk,
                    'accuracy': accuracy,
                    'mse': mse,
                    'mae': mae,
                    'samples_used': samples_used if samples_used is not None else total_samples,
                    'total_samples': total_samples,
                    'model': model,
                })
        except Exception:
            continue
    
    # Validate collected results
    if not all_results:
        if model_hf_id:
            print(f"❌ No experiments found for model: {model_hf_id}")
            print("\nAvailable models:")
            _show_available_models(exp_base)
        else:
            print("❌ No valid experiments found.")
        return None

    df = pd.DataFrame(all_results)
    
    # Get best QWK for each category/template/output combination
    best_results = df.loc[df.groupby(['category', 'template', 'output_format'])['qwk'].idxmax()]
    return best_results


def _show_available_models(exp_base: Path) -> None:
    """Show available models in experiments."""
    all_models = set()
    for category_file in exp_base.glob("*.json"):
        try:
            with open(category_file, 'r', encoding='utf-8') as f:
                experiments = json.load(f)
            for exp in experiments:
                all_models.add(exp.get('model', 'unknown'))
        except Exception:
            continue
    
    if all_models:
        for m in sorted(all_models):
            print(f"  • {m}")
    else:
        print("  (No model information found)")


def _create_bar_chart(
    best_results: pd.DataFrame,
    model_hf_id: Optional[str],
    figsize: tuple
) -> None:
    """Create grouped bar chart visualization with custom ordering."""
    # Filter to only combinations that exist in data
    existing_combos = set(zip(best_results['template'], best_results['output_format']))
    ordered_combos = [combo for combo in COMBO_ORDER if combo in existing_combos]
    
    # Create pivot with ordered columns
    pivot = best_results.pivot_table(
        values='qwk',
        index='category',
        columns=['template', 'output_format'],
        aggfunc='max'
    )
    
    # Reorder columns
    available_cols = [col for col in ordered_combos if col in pivot.columns]
    pivot = pivot[available_cols]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot grouped bars with spacing between categories
    n_categories = len(pivot.index)
    n_bars = len(pivot.columns)
    
    # Increased spacing between categories (1.5 gives 50% extra space)
    x = np.arange(n_categories) * 1.5
    width = 1.3 / n_bars  # Bar width
    
    for i, col in enumerate(pivot.columns):
        offset = width * (i - n_bars/2 + 0.5)
        template, output_fmt = col
        label = TEMPLATE_LABELS.get(col, f"{template.replace('_', ' ').title()} ({output_fmt.upper()})")
        values = pivot[col].values
        color = COLOR_MAP.get(col, f'C{i}')
        
        bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.85)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=7
                )
    
    # Customize plot
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best QWK Score', fontsize=12, fontweight='bold')
    
    title = 'Best Quadratic Weighted Kappa by Category'
    if model_hf_id:
        model_short = model_hf_id.split('/')[-1]
        title += f'\nModel: {model_short}'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha='right')
    ax.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)
    
    # Adjust x-axis limits to accommodate spacing
    ax.set_xlim(-0.8, x[-1] + 0.8)
    ax.margins(x=0)
    
    plt.tight_layout()
    plt.show()


def _print_summary_statistics(
    best_results: pd.DataFrame,
    model_hf_id: Optional[str]
) -> pd.DataFrame:
    """Print summary statistics and best result."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    if model_hf_id:
        print(f"Model: {model_hf_id}")
    print("="*80)
    
    # Group and aggregate
    summary = best_results.groupby(['template', 'output_format'])['qwk'].agg(
        ['mean', 'median', 'min', 'max', 'count']
    )
    
    weights = pd.to_numeric(best_results.get('samples_used'), errors='coerce')
    if weights.notna().any():
        weighted_df = best_results.copy()
        weighted_df['weight'] = weights
        weighted_df['qwk_weighted'] = pd.to_numeric(weighted_df['qwk'], errors='coerce') * weighted_df['weight']
        weighted_df['accuracy_weighted'] = pd.to_numeric(weighted_df['accuracy'], errors='coerce') * weighted_df['weight']
        weighted_df['mse_weighted'] = pd.to_numeric(weighted_df['mse'], errors='coerce') * weighted_df['weight']
        weighted_df['mae_weighted'] = pd.to_numeric(weighted_df['mae'], errors='coerce') * weighted_df['weight']

        valid_mask = weighted_df['weight'].notna() & (weighted_df['weight'] > 0)
        valid_weighted = weighted_df[valid_mask]

        grouped = valid_weighted.groupby(['template', 'output_format'], dropna=False).agg(
            weight_sum=('weight', 'sum'),
            qwk_weighted_sum=('qwk_weighted', 'sum'),
            accuracy_weighted_sum=('accuracy_weighted', 'sum'),
            mse_weighted_sum=('mse_weighted', 'sum'),
            mae_weighted_sum=('mae_weighted', 'sum'),
        )
        summary['Weighted QWK'] = grouped['qwk_weighted_sum'] / grouped['weight_sum']
        summary['Weighted Accuracy'] = grouped['accuracy_weighted_sum'] / grouped['weight_sum']
        summary['Weighted MSE'] = grouped['mse_weighted_sum'] / grouped['weight_sum']
        summary['Weighted MAE'] = grouped['mae_weighted_sum'] / grouped['weight_sum']
    else:
        summary['Weighted QWK'] = np.nan
        summary['Weighted Accuracy'] = np.nan
        summary['Weighted MSE'] = np.nan
        summary['Weighted MAE'] = np.nan

    summary = summary.rename(
        columns={
            'mean': 'Mean QWK',
            'median': 'Median QWK',
            'min': 'Min QWK',
            'max': 'Max QWK',
            'count': 'Categories',
        }
    )

    summary = summary.reset_index()
    summary = summary.reindex(
        columns=[
            'template',
            'output_format',
            'Mean QWK',
            'Median QWK',
            'Min QWK',
            'Max QWK',
            'Weighted QWK',
            'Weighted Accuracy',
            'Weighted MSE',
            'Weighted MAE',
            'Categories',
        ]
    )

    # Sort using the same order as the chart
    order_dict = {combo: i for i, combo in enumerate(COMBO_ORDER)}
    summary['sort_order'] = summary.apply(
        lambda row: order_dict.get((row['template'], row['output_format']), 99),
        axis=1,
    )
    summary = summary.sort_values('sort_order').drop('sort_order', axis=1)
    summary = summary.reset_index(drop=True)
    
    return summary


def _plot_summary_panels(
    summary: pd.DataFrame,
    model_hf_id: Optional[str],
    figsize: tuple
) -> None:
    ordered_combos = [
        combo for combo in COMBO_ORDER
        if not summary[(summary['template'] == combo[0])
                       & (summary['output_format'] == combo[1])].empty
    ]

    if not ordered_combos:
        print("❌ No summary rows to plot.")
        return

    labels = [TEMPLATE_LABELS.get(c, f"{c[0].replace('_', ' ').title()} ({c[1].upper()})")
              for c in ordered_combos]
    colors = [COLOR_MAP.get(c, f"C{i}") for i, c in enumerate(ordered_combos)]

    metric_cols = [
        ("Median QWK", "Median QWK"),
        ("Weighted QWK", "Weighted QWK"),
        ("Weighted Accuracy", "Weighted Accuracy"),
        ("Weighted MSE", "Weighted MSE (lower is better)"),
        ("Weighted MAE", "Weighted MAE (lower is better)"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for ax, (metric_key, title) in zip(axes, metric_cols):
        values = []
        for combo in ordered_combos:
            row = summary[(summary['template'] == combo[0])
                          & (summary['output_format'] == combo[1])]
            values.append(float(row[metric_key].iloc[0]) if not row.empty else np.nan)

        x = np.arange(len(ordered_combos))
        bars = ax.bar(x, values, color=colors, alpha=0.85)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        if metric_key in {"Weighted QWK", "Weighted Accuracy", "Median QWK"}:
            ax.set_ylim(0, 1.0)
        else:
            max_val = np.nanmax(values) if np.isfinite(values).any() else 1.0
            ax.set_ylim(0, max_val * 1.15)

        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha='center',
                    va='bottom',
                    fontsize=7
                )

    for ax in axes[len(metric_cols):]:
        ax.set_visible(False)

    title = "Summary Statistics"
    if model_hf_id:
        title += f"\nModel: {model_hf_id.split('/')[-1]}"
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
