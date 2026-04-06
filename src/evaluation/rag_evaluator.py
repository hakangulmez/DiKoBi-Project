"""
RAG evaluation for single categories.

This module provides functions for evaluating RAG-based classification
on a single category, including both RAG Balanced and RAG Similarity methods.
Results are logged to ExperimentTracker for unified history tracking.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time

import pandas as pd

from src.rag import RAGClassifier
from src.utils import ExperimentTracker
from src.evaluation.metrics import calculate_metrics
from src.models import get_model_config


def build_predictions_list(
    texts: List[str],
    true_ratings: List[int],
    predictions: List[Optional[int]],
    raw_outputs: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Build predictions list for ExperimentTracker.
    
    Args:
        texts: Input texts
        true_ratings: True rating values
        predictions: Predicted rating values (None if parsing failed)
        raw_outputs: Raw model outputs (optional)
    
    Returns:
        List of prediction dictionaries for tracker
    """
    predictions_list = []
    for i, (text, y_true, y_pred) in enumerate(zip(texts, true_ratings, predictions)):
        predictions_list.append({
            "sample_index": i,
            "text": text[:200] + "..." if len(text) > 200 else text,
            "y_true": y_true,
            "y_pred": y_pred,
            "parse_ok": y_pred is not None,
            "raw_output": raw_outputs[i] if raw_outputs else "",
        })
    return predictions_list


def evaluate_rag_single_category(
    category: str,
    model_hf_id: str,
    rag_index_path: Path,
    test_dir: Path,
    output_format: str,
    max_samples: Optional[int] = None,
    use_8bit: bool = True,
    max_new_tokens: int = 15,
    n_examples: int = 3,
    batch_size: int = 2,
    temperature: float = 0.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate RAG-based classification for a single category.
    
    Tests both RAG Balanced and RAG Similarity methods.
    Results are automatically saved to ExperimentTracker.
    
    Args:
        category: Category to evaluate (e.g., "1_D_M")
        model_hf_id: HuggingFace model ID
        rag_index_path: Path to RAG index
        test_dir: Directory containing test CSV files
        output_format: Output format - "json" or "text" (required, determined from model config)
        max_samples: Maximum number of test samples (None = all)
        use_8bit: Use 8-bit quantization
        max_new_tokens: Maximum tokens for generation (from notebook config)
        n_examples: Number of RAG examples to retrieve
        batch_size: Batch size for classification
        temperature: Sampling temperature
        verbose: Print progress messages
    
    Returns:
        Dictionary with results for each method:
        {
            "category": str,
            "model": str,
            "results": {
                "RAG Balanced": {
                    "predictions": List[int],
                    "metrics": Dict[str, float],
                    "time": float,
                    "cached": bool
                },
                "RAG Similarity": {...}
            }
        }
    """
    if verbose:
        print("=" * 70)
        print(f"RAG EVALUATION: {category}")
        print("=" * 70)
        print(f"Model:        {model_hf_id}")
        print(f"Test samples: {max_samples or 'Full test set'}")
        print(f"RAG examples: {n_examples}")
        print(f"Batch size:   {batch_size}")
        print(f"Temperature:  {temperature}")
        print("=" * 70)
    
    # Check prerequisites
    if not rag_index_path.exists():
        raise FileNotFoundError(f"RAG index not found: {rag_index_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test data not found: {test_dir}")
    
    # Load test data
    test_file = test_dir / f"{category}.csv"
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    test_df = pd.read_csv(test_file)
    test_df = test_df[test_df[category].notna()].copy()
    total_samples = len(test_df)
    
    if max_samples:
        test_df = test_df.head(max_samples)
    
    texts = test_df["free_text_answer"].tolist()
    true_ratings = test_df[category].astype(int).tolist()
    samples_used = len(texts)
    
    if verbose:
        print(f"\n📊 Loaded {samples_used} test samples")
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(category)
    
    # RAG methods to test
    rag_methods = [
        ("RAG Balanced", "rag_balanced", True),
        ("RAG Similarity", "rag_similarity", False),
    ]
    
    classifier_loaded = False
    results = {
        "category": category,
        "model": model_hf_id,
        "results": {}
    }
    
    try:
        for idx, (method_name, template_key, use_balanced) in enumerate(rag_methods, 1):
            if verbose:
                print("\n" + "-" * 70)
                print(f"Running: {method_name} ({idx}/{len(rag_methods)})")
                print("-" * 70)
            
            # Check for duplicate experiment
            check_result = tracker.check_duplicate(
                model=model_hf_id,
                prompt_template=template_key,
                max_samples=max_samples,
                use_8bit=use_8bit,
                max_new_tokens=max_new_tokens,
                output_format=output_format,  # Pass output format explicitly
                verbose=verbose,
            )
            
            if check_result['cached']:
                # Use existing results
                cached = check_result['cached']
                metrics = cached.get('metrics', {})
                
                # Reconstruct predictions from cached experiment
                cached_preds = [None] * samples_used
                if 'predictions' in cached:
                    for pred in cached['predictions']:
                        idx_p = pred.get('sample_index')
                        if idx_p is not None and idx_p < len(cached_preds):
                            cached_preds[idx_p] = pred.get('y_pred')
                
                results["results"][method_name] = {
                    "predictions": cached_preds,
                    "metrics": metrics,
                    "time": 0,
                    "cached": True
                }
                continue
            
            # No duplicate - run classification
            # Lazy load classifier only when needed
            if not classifier_loaded:
                if verbose:
                    print("   Loading RAG classifier...")
                rag_classifier = RAGClassifier(
                    model_name=model_hf_id,
                    retriever_path=str(rag_index_path),
                    use_8bit=use_8bit,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    batch_size=batch_size,
                )
                classifier_loaded = True
            
            # Run classification
            start = time.time()
            rag_results = rag_classifier.classify_batch(
                category=category,
                texts=texts,
                n_examples=n_examples,
                use_balanced=use_balanced,
                return_raw=True
            )
            elapsed_time = time.time() - start
            
            preds = [r["score"] for r in rag_results]
            raw_outputs = [r.get("raw_output", "") for r in rag_results]
            
            # Calculate metrics
            valid_pairs = [(t, p) for t, p in zip(true_ratings, preds) if p is not None]
            if valid_pairs:
                valid_true, valid_pred = zip(*valid_pairs)
                metrics = calculate_metrics(list(valid_true), list(valid_pred))
            else:
                metrics = {"qwk": 0, "mse": float('inf'), "accuracy": 0, "mae": 0}
            
            samples_processed = len(valid_pairs)
            
            # Build predictions list for tracker
            predictions_list = build_predictions_list(texts, true_ratings, preds, raw_outputs)
            
            # Build prompt text description
            if template_key == "rag_balanced":
                prompt_text = f"[RAG Balanced] Dynamic examples from retriever (n={n_examples}, balanced by rating)"
            else:  # rag_similarity
                prompt_text = f"[RAG Similarity] Dynamic examples from retriever (n={n_examples}, similarity-based)"
            
            # Log to ExperimentTracker
            tracker.log_experiment(
                model=model_hf_id,
                prompt_template=template_key,
                prompt_text=prompt_text,
                metrics=metrics,
                dataset_path=str(test_file),
                total_samples=total_samples,
                samples_used=samples_processed,
                max_samples=max_samples,
                quantization="8bit" if use_8bit else "none",
                output_format=output_format,
                generation_params={
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": 1.0,
                    "do_sample": False,
                    "batch_size": batch_size,
                    "n_examples": n_examples,
                    "balanced": use_balanced
                },
                predictions=predictions_list,
            )
            
            # Store results
            results["results"][method_name] = {
                "predictions": preds,
                "metrics": metrics,
                "time": elapsed_time,
                "cached": False
            }
            
            if verbose:
                samples_skipped = samples_used - samples_processed
                print(
                    f"   ✓ Completed in {elapsed_time:.1f}s"
                    + (f" ({samples_skipped} failed parses)" if samples_skipped else "")
                )
    
    finally:
        # Unload classifier if it was loaded
        if classifier_loaded:
            try:
                rag_classifier.unload()
            except:
                pass
    
    return results


def print_results_summary(results: Dict[str, Any]) -> None:
    """
    Print a formatted summary of RAG evaluation results.
    
    Args:
        results: Results dictionary from evaluate_rag_single_category
    """
    print("\n" + "=" * 70)
    print("RAG RESULTS SUMMARY")
    print("=" * 70)
    
    if not results.get("results"):
        print("\n⚠️  No results to display.")
        return
    
    comparison_data = []
    
    for method_name, data in results["results"].items():
        preds = data["predictions"]
        metrics = data["metrics"]
        
        # Count failed predictions
        failed = sum(1 for p in preds if p is None)
        
        comparison_data.append({
            "Method": method_name,
            "QWK": metrics.get("qwk", 0),
            "MSE": metrics.get("mse", float('inf')),
            "Accuracy": metrics.get("accuracy", 0),
            "Time (s)": data["time"],
            "Failed": failed,
            "Cached": "✓" if data.get("cached") else ""
        })
    
    # Create comparison table
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values("QWK", ascending=False, na_position="last")
    comparison_df.insert(0, "Rank", range(1, len(comparison_df) + 1))
    
    print(comparison_df.to_string(index=False))
    
    # Highlight winner
    best = comparison_df.iloc[0]
    print("\n" + "=" * 70)
    print(f"🏆 BEST RAG METHOD: {best['Method']}")
    print("=" * 70)
    print(f"   MSE:      {best['MSE']:.4f}")
    print(f"   Accuracy: {best['Accuracy']:.4f}")
    print(f"   QWK:      {best['QWK']:.4f}")
    print(f"   Time:     {best['Time (s)']:.1f}s")
    print("=" * 70)
