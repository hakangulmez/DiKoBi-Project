"""Model comparison tool.

Compare multiple LLM models on the same dataset to find the best one.
Corresponds to Phase 3 of the development workflow.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Dict, List
from datetime import datetime
import json
import logging
import time
import gc

from ..classification import TextClassifier
from ..classification.prompts import get_prompt
from ..models import get_compatible_models, get_model_config, MODELS
from .metrics import calculate_metrics, compare_metrics
from ..utils import ExperimentTracker

logger = logging.getLogger(__name__)


class ModelComparison:
    """
    Compare multiple LLM models on classification task.
    
    This is Phase 3 of the workflow: model comparison using best prompt from Phase 2.
    
    Example:
        comparison = ModelComparison(
            category="1_D_M",
            test_data_path="data/processed/test/1_D_M.csv",
            prompt_template="zero_shot",
            max_samples=None
        )
        
        results = comparison.compare_all_models()
        comparison.save_report("data/processed/model_comparison/")
    """
    
    def __init__(
        self,
        category: str,
        test_data_path: str,
        prompt_template: str = "zero_shot",
        max_samples: Optional[int] = None,
        use_8bit: bool = False,
        show_progress: bool = False,
    ):
        """
        Initialize model comparison.
        
        Args:
            category: DiKoBi category to test
            test_data_path: Path to CSV with test data
            prompt_template: Prompt template to use (from Phase 1 or 2)
            max_samples: Limit number of samples (None = all)
            use_8bit: Use 8-bit quantization
        """
        self.category = category
        self.test_data_path = test_data_path
        self.prompt_template = prompt_template
        self.max_samples = max_samples
        self.use_8bit = use_8bit
        self.show_progress = show_progress
        
        # Load test data
        logger.info(f"Loading test data from {test_data_path}")
        df_full = pd.read_csv(test_data_path)
        self.total_samples = int(len(df_full))
        self.df = df_full
        
        if max_samples:
            self.df = self.df.head(max_samples)
            logger.info(f"Limited to {max_samples} samples")
        
        logger.info(f"Test set size: {len(self.df)}")
        logger.info(f"Label distribution:\n{self.df[category].value_counts().sort_index()}")
        
        # Results storage
        self.results: Dict[str, Dict] = {}
        self._current_classifier = None
    
    def _cleanup_model(self):
        """Clean up model resources between tests with aggressive memory clearing."""
        import gc
        import torch
        from ..utils.device import get_device
        
        # Clear any classifier instance
        if hasattr(self, '_current_classifier'):
            if self._current_classifier is not None:
                try:
                    self._current_classifier.unload()
                except Exception as e:
                    logger.warning(f"Error unloading classifier: {e}")
                del self._current_classifier
                self._current_classifier = None
        
        # Force garbage collection (multiple passes for thorough cleanup)
        gc.collect()
        gc.collect()
        
        # Clear GPU cache aggressively
        device = get_device()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # Second pass
        elif device.type == "mps":
            # Aggressive MPS cleanup to prevent OOM
            torch.mps.empty_cache()
            gc.collect()  # Second pass
            torch.mps.empty_cache()  # Clear again
            gc.collect()  # Final pass
        
        # Final garbage collection
        gc.collect()
    
    def compare_all_models(self, models: Optional[List[str]] = None) -> Dict:
        """
        Compare all compatible models.
        
        Args:
            models: List of model keys to test (None = all compatible)
        
        Returns:
            Dictionary mapping model keys to results
        """
        from ..models import MODELS
        
        if models is None:
            # Auto-detect compatible models
            models = get_compatible_models(
                use_8bit=self.use_8bit
            )
            logger.info(f"Testing {len(models)} compatible models: {models}")
        
        # Deduplicate models list (preserve order)
        seen = set()
        unique_models = []
        for m in models:
            if m not in seen:
                seen.add(m)
                unique_models.append(m)
        
        if len(unique_models) != len(models):
            duplicates = len(models) - len(unique_models)
            logger.warning(f"⚠️  Removed {duplicates} duplicate model(s) from test list")
            print(f"⚠️  Removed {duplicates} duplicate model(s) from test list")
        
        models = unique_models
        
        if not models:
            raise ValueError("No compatible models found for your hardware!")
        
        # ===========================
        # PHASE 1: Check what work is needed (no model loading!)
        # ===========================
        print("\n" + "="*70)
        print("PHASE 1: Checking experiment cache...")
        print("="*70)
        
        tracker = ExperimentTracker(self.category)
        models_to_test = []  # Models that need testing
        cached_results = []  # Models with cached results
        
        for model_key in models:
            check_result = tracker.check_duplicate(
                model=model_key,
                prompt_template=self.prompt_template,
                max_samples=self.max_samples,
                use_8bit=self.use_8bit,
                verbose=False,
            )
            
            if check_result['cached']:
                cached_results.append((model_key, check_result['cached']))
                print(f"✓ {model_key}: CACHED")
            else:
                models_to_test.append(model_key)
                print(f"○ {model_key}: NEEDS RUN")
        
        print()
        print(f"Cache scan complete: {len(cached_results)} cached, {len(models_to_test)} to run")
        print()
        
        # Add cached results to self.results
        for model_key, cached in cached_results:
            model_config = get_model_config(model_key)
            metrics = cached.get('metrics', {})
            gen_params = cached.get('generation_params', {})
            
            self.results[model_key] = {
                "model_key": model_key,
                "model_name": model_config.name,
                "model_hf_id": model_config.hf_id,
                "model_params": model_config.params,
                "metrics": metrics,
                "y_true": [],  # Don't need full arrays for cached
                "y_pred": [],
                "time_seconds": 0.0,
                "load_time": 0.0,
                "predict_time": 0.0,
                "time_per_sample": None,
                "samples_processed": cached['dataset']['samples_used'],
                "samples_skipped": 0,
                "failed_parses": [],
                "from_cache": True,
                "experiment_id": cached.get('experiment_id'),
                "generation_params": gen_params,
            }
        
        # If all models cached, skip testing phase entirely
        if not models_to_test:
            print("🎉 All models already cached! No model loading needed.")
            return self.results
        
        # ===========================
        # PHASE 2: Load and test models
        # ===========================
        print("="*70)
        print("PHASE 2: Loading and testing models...")
        print("="*70)
        print()
        
        total_models = len(models_to_test)
        for idx, model_key in enumerate(models_to_test, 1):
            # Aggressive cleanup BEFORE loading new model
            self._cleanup_model()
            
            # Give system time to release memory (helps on Windows)
            time.sleep(0.5)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing model [{idx}/{total_models}]: {model_key}")
            logger.info(f"{'='*60}")
            
            print(f"\n[{idx}/{total_models}] Testing {model_key}...")
            
            try:
                result = self._test_single_model(model_key)
                self.results[model_key] = result
                
                # Print quick summary (should never be from_cache since we filtered)
                max_tok = result.get('generation_params', {}).get('max_new_tokens', 'N/A')
                print(f"✓ {model_key} complete:")
                print(f"  • Accuracy: {result['metrics']['accuracy']:.2%}")
                print(f"  • QWK: {result['metrics']['qwk']:.3f}")
                print(f"  • MaxTokens: {max_tok}")
                print(f"  • Time: {result['time_seconds']:.1f}s")
                
            except Exception as e:
                logger.error(f"Failed to test {model_key}: {e}")
                self.results[model_key] = {"error": str(e)}
                print(f"  ✗ Error: {e}")
        
        return self.results
    
    def test_single_model(self, model_key: str) -> Dict:
        """
        Test a single model on the dataset.
        
        Public method for testing one model at a time.
        Useful for incremental testing with checkpointing.
        
        Args:
            model_key: Model key from registry (e.g., 'qwen-1.5b')
        
        Returns:
            Dictionary with model results and metrics
        """
        return self._test_single_model(model_key)

    def test_single_model_json(self, model_key: str, include_predictions: bool = False) -> Dict[str, Any]:
        """Test a single model and return a JSON-serializable result dict."""
        result = self._test_single_model(model_key)
        return self._make_jsonable(result, include_predictions=include_predictions)
    
    def _test_single_model(self, model_key: str) -> Dict:
        """Test a single model on the dataset (internal implementation).
        
        Note: When called from compare_all_models(), cache check already done in Phase 1.
        When called standalone (test_single_model()), cache check done here.
        """
        model_config = get_model_config(model_key)
        
        # Extract texts before any processing
        texts = self.df["free_text_answer"].astype(str).tolist()
        true_labels = self.df[self.category].tolist()
        
        # Get tracker and check result for metadata
        tracker = ExperimentTracker(self.category)
        
        check_result = tracker.check_duplicate(
            model=model_key,
            prompt_template=self.prompt_template,
            max_samples=self.max_samples,
            use_8bit=self.use_8bit,
            verbose=True,
        )
        
        # Check if cached (for standalone test_single_model() calls)
        cached = check_result['cached']
        if cached:
            # DUPLICATE FOUND - Return cached results without loading model
            metrics = cached.get('metrics', {})
            gen_params = cached.get('generation_params', {})
            return {
                "model_key": model_key,
                "model_name": model_config.name,
                "model_hf_id": model_config.hf_id,
                "model_params": model_config.params,
                "metrics": metrics,
                "y_true": [],  # Don't need full arrays for duplicate
                "y_pred": [],
                "time_seconds": 0.0,  # Cached, no time spent
                "load_time": 0.0,
                "predict_time": 0.0,
                "time_per_sample": None,
                "samples_processed": cached['dataset']['samples_used'],
                "samples_skipped": 0,
                "failed_parses": [],
                "from_cache": True,
                "experiment_id": cached.get('experiment_id'),
                "generation_params": gen_params,  # Include for reproducibility
            }
        
        # NO DUPLICATE - Proceed with actual testing
        # Initialize classifier
        start_time = time.time()
        classifier = TextClassifier(
            model_name=model_key,  # Pass model_key to get correct format config
            use_8bit=self.use_8bit,
            batch_size=2,
        )
        self._current_classifier = classifier
        load_time = time.time() - start_time
        
        # Run predictions
        predict_start = time.time()
        
        results = classifier.classify_batch(
            category=self.category,
            texts=texts,
            prompt_template=self.prompt_template,
            show_progress=self.show_progress,
            return_raw=True  # Always capture raw outputs for verification
        )

        predictions = []
        for i, r in enumerate(results):
            y_pred_i = r.get("score")
            predictions.append(
                {
                    "sample_index": i,
                    "text": texts[i],
                    "y_true": true_labels[i],
                    "y_pred": y_pred_i,
                    "parse_ok": y_pred_i is not None,
                    "is_correct": y_pred_i == true_labels[i] if y_pred_i is not None else False,
                    "raw_output": r.get("raw_output", "")
                }
            )
        
        predict_time = time.time() - predict_start
        total_time = time.time() - start_time
        
        # Extract valid predictions and collect parsing failures
        y_pred = [r['score'] for r in results if r['score'] is not None]
        y_true = [true_labels[i] for i, r in enumerate(results) if r['score'] is not None]
        
        # Collect failed parses for debugging
        failed_parses = [
            {
                "index": i,
                "text": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                "raw_output": r.get('raw_output', 'N/A')[:200],
                "expected_range": r['valid_range']
            }
            for i, r in enumerate(results)
            if r['score'] is None
        ]
        
        # Calculate metrics
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # Calculate processed samples before using it
        samples_processed = int(len(y_pred))
        samples_skipped = len(texts) - samples_processed
        time_per_sample = float(predict_time / samples_processed) if samples_processed > 0 else None
        
        # Print warning if many failures
        if samples_skipped > 0:
            skip_rate = samples_skipped / len(texts)
            if skip_rate > 0.1:  # More than 10% failed
                logger.warning(f"⚠️  {samples_skipped}/{len(texts)} samples failed to parse ({skip_rate:.1%})")
                logger.warning("   Consider refining prompts or checking parser logic")
        
        # Capture generation params before deleting classifier
        final_generation_params = classifier.get_generation_params()
        
        # Log experiment (+ predictions CSV)
        # Use metadata computed by check_duplicate
        tracker.log_experiment(
            model=check_result['model_hf_id'],
            prompt_template=self.prompt_template,
            prompt_text=check_result['prompt_text'],
            metrics=metrics,
            dataset_path=check_result['dataset_path'],
            total_samples=self.total_samples,
            samples_used=samples_processed,
            max_samples=self.max_samples,
            quantization=check_result['quantization'],
            output_format=check_result['output_format'],
            generation_params=final_generation_params,
            version_name=f"model_{model_key}",
            predictions=predictions,
        )

        # Clear GPU memory
        classifier.unload()
        del classifier
        self._cleanup_model()

        return {
            "model_key": model_key,
            "model_name": model_config.name,
            "model_hf_id": model_config.hf_id,
            "model_params": model_config.params,
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": y_pred,
            "time_seconds": total_time,
            "load_time": load_time,
            "predict_time": predict_time,
            "time_per_sample": time_per_sample,
            "samples_processed": samples_processed,
            "samples_skipped": samples_skipped,
            "failed_parses": failed_parses[:5] if failed_parses else [],  # Keep first 5 for report
            "generation_params": final_generation_params,  # Include for reproducibility
        }

    @staticmethod
    def _make_jsonable(obj: Any, include_predictions: bool = False) -> Any:
        """Convert nested structures (numpy, Path, datetime) to JSON-serializable types."""
        if isinstance(obj, dict):
            converted: Dict[str, Any] = {}
            for k, v in obj.items():
                if not include_predictions and k in {"y_true", "y_pred"}:
                    continue
                converted[k] = ModelComparison._make_jsonable(v, include_predictions=include_predictions)
            return converted

        if isinstance(obj, (list, tuple)):
            return [ModelComparison._make_jsonable(v, include_predictions=include_predictions) for v in obj]

        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, (np.integer,)):
            return int(obj)

        if isinstance(obj, (np.floating,)):
            return float(obj)

        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        # sklearn confusion matrix is often a numpy ndarray; handled above.
        return obj
    
    def show_parsing_failures(self, model_key: Optional[str] = None):
        """Show detailed information about parsing failures.
        
        Args:
            model_key: Show failures for specific model (None = worst model)
        """
        if not self.results:
            print("No results available.")
            return
        
        # If no model specified, find the one with most failures
        if model_key is None:
            model_key = max(
                (k for k, v in self.results.items() if 'error' not in v),
                key=lambda k: self.results[k].get('samples_skipped', 0),
                default=None
            )
        
        if model_key not in self.results:
            print(f"Model '{model_key}' not found in results.")
            return
        
        result = self.results[model_key]
        if 'error' in result:
            print(f"Model '{model_key}' had an error: {result['error']}")
            return
        
        failed_parses = result.get('failed_parses', [])
        samples_skipped = result.get('samples_skipped', 0)
        
        print("\n" + "="*80)
        print(f"PARSING FAILURES: {model_key}")
        print("="*80)
        print(f"Total failed: {samples_skipped}")
        
        if not failed_parses:
            print("✓ No parsing failures (or details not captured)")
        else:
            print(f"\nShowing first {len(failed_parses)} examples:\n")
            for i, failure in enumerate(failed_parses, 1):
                print(f"[{i}] Index {failure['index']}")
                print(f"    Input: {failure['text']}")
                print(f"    Output: {failure['raw_output']}")
                print(f"    Expected: one of {failure['expected_range']}")
                print()
        
        print("="*80 + "\n")
    
    def print_summary(self):
        """Print comparison summary."""
        if not self.results:
            print("No results to display. Run compare_all_models() first.")
            return
        
        # Filter out errors
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not valid_results:
            print("No successful results.")
            return
        
        # Sort by QWK
        sorted_results = sorted(
            valid_results.items(),
            key=lambda x: x[1]['metrics']['qwk'],
            reverse=True
        )
        
        print("\n" + "=" * 100)
        print("MODEL COMPARISON RESULTS")
        print("=" * 100)
        print(f"Category: {self.category}")
        print(f"Prompt: {self.prompt_template}")
        print(f"Samples: {len(self.df)}")
        print("\n" + "-" * 100)
        print(f"{'Rank':<6} {'Model':<25} {'Params':<8} {'MaxTok':<8} {'Accuracy':<12} {'QWK':<12} {'Time (s)':<12} {'Failed':<8}")
        print("-" * 100)
        
        # Get max_tokens from first result if available
        sample_result = next(iter(valid_results.values()))
        
        for rank, (key, result) in enumerate(sorted_results, 1):
            m = result['metrics']
            skipped = result.get('samples_skipped', 0)
            marker = "⭐" if rank == 1 else "  "
            skip_str = f"{skipped}" if skipped > 0 else "-"
            skip_warn = " ⚠️" if skipped > len(self.df) * 0.1 else ""  # Warn if >10% failed
            
            # Get max_tokens - check result first, then try to get from tracker
            max_tokens = "N/A"
            if hasattr(result, '__getitem__'):
                gen_params = result.get('generation_params', {})
                if gen_params and 'max_new_tokens' in gen_params:
                    max_tokens = str(gen_params['max_new_tokens'])
            
            print(f"{marker} {rank:<4} {key:<25} {result['model_params']:<8} "
                  f"{max_tokens:<8} {m['accuracy']:<12.2%} {m['qwk']:<12.4f} {result['time_seconds']:<12.1f} {skip_str:<8}{skip_warn}")
        
        print("=" * 100)
        
        # Print recommendation
        best = sorted_results[0][1]
        best = sorted_results[0][1]
        print(f"\n✓ RECOMMENDED: {best['model_name']} (QWK: {best['metrics']['qwk']:.4f})")
        
        # Warn about parsing failures
        high_failure_models = [
            (key, r) for key, r in valid_results.items()
            if r.get('samples_skipped', 0) > len(self.df) * 0.1
        ]
        if high_failure_models:
            print(f"\n⚠️  Warning: {len(high_failure_models)} model(s) had >10% parsing failures:")
            for key, r in high_failure_models:
                skip_rate = r['samples_skipped'] / len(self.df)
                print(f"    • {r['model_name']}: {r['samples_skipped']} failed ({skip_rate:.1%})")
            print("    Use comparison.show_parsing_failures('<model_key>') to see examples")
        
        print()
    
    def save_report(self, output_dir: str):
        """Save comparison results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed report
        report_path = output_path / f"model_comparison_{self.category}_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Category: {self.category}\n")
            f.write(f"Prompt Template: {self.prompt_template}\n")
            f.write(f"Test Samples: {len(self.df)}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Sort by QWK score
            sorted_results = sorted(
                [(k, v) for k, v in self.results.items() if 'error' not in v],
                key=lambda x: x[1]['metrics']['qwk'],
                reverse=True
            )
            
            f.write("RANKINGS (by QWK):\n")
            f.write("-" * 80 + "\n")
            
            for rank, (model_key, result) in enumerate(sorted_results, 1):
                f.write(f"\n{rank}. {result['model_name']} ({model_key})\n")
                f.write(f"   Parameters:  {result['model_params']}\n")
                f.write(f"   QWK:         {result['metrics']['qwk']:.4f}\n")
                f.write(f"   Accuracy:    {result['metrics']['accuracy']:.2%}\n")
                f.write(f"   MSE:         {result['metrics']['mse']:.4f}\n")
                f.write(f"   MAE:         {result['metrics']['mae']:.4f}\n")
                f.write(f"   Time:        {result['time_seconds']:.1f}s\n")
                f.write(f"   Speed:       {result['samples_processed']/result['predict_time']:.1f} samples/sec\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"RECOMMENDATION: {sorted_results[0][1]['model_name']}\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Report saved to {report_path}")
        
        # Save CSV with results
        csv_path = output_path / f"model_comparison_{self.category}_{timestamp}.csv"
        
        rows = []
        for model_key, result in self.results.items():
            if 'error' not in result:
                rows.append({
                    'model_key': model_key,
                    'model_name': result['model_name'],
                    'model_params': result['model_params'],
                    'qwk': result['metrics']['qwk'],
                    'accuracy': result['metrics']['accuracy'],
                    'mse': result['metrics']['mse'],
                    'mae': result['metrics']['mae'],
                    'time_seconds': result['time_seconds'],
                    'samples_per_sec': result['samples_processed'] / result['predict_time'],
                    'samples_processed': result['samples_processed']
                })
        
        df_results = pd.DataFrame(rows)
        df_results = df_results.sort_values('qwk', ascending=False)
        df_results.to_csv(csv_path, index=False)
        logger.info(f"CSV saved to {csv_path}")


@dataclass(frozen=True)
class ModelComparisonConfig:
    """Configuration for model comparison (Phase 3)."""

    category: str = "1_D_M"
    test_data_path: Optional[str] = None

    # If None, Phase 1 recommendation is used (if available), else `default_template`.
    prompt_template: Optional[str] = None
    default_template: str = "zero_shot"

    max_samples: Optional[int] = None

    # Quantization: Always use 8-bit by default for stability
    use_8bit: bool = True

    # Output
    output_dir: str = "data/processed/model_comparison"

    # Notebook stability
    show_progress: bool = False
    
    # Use JSON output format for better parsing (more reliable than text format)
    use_json: bool = True


def _resolve_prompt_template(prompt_template: Optional[str], default_template: str) -> str:
    """Resolve prompt template using explicitly set value or default."""
    
    if prompt_template:
        print(f"✅ Using template: {prompt_template}")
        return prompt_template
    
    print(f"✅ Using default template: {default_template}")
    return default_template


def compare_all_models(config: Optional[ModelComparisonConfig] = None, **overrides: Any) -> Dict[str, Any]:
    """Run model comparison (Phase 3) and save a report.

    Args:
        config: Optional ModelComparisonConfig object. If omitted, defaults are used.
        **overrides: Optional keyword overrides for ModelComparisonConfig fields.

    Returns:
        Results dict from ModelComparison.compare_all_models().
    """
    from ..models import MODELS

    base = config or ModelComparisonConfig()

    # Apply overrides over base config
    category = overrides.get("category") or base.category
    test_data_path = overrides.get("test_data_path") or base.test_data_path or f"data/processed/test/{category}.csv"
    max_samples = overrides.get("max_samples") if "max_samples" in overrides else base.max_samples
    prompt_template = overrides.get("prompt_template") if "prompt_template" in overrides else base.prompt_template
    default_template = overrides.get("default_template") or base.default_template
    output_dir = overrides.get("output_dir") or base.output_dir
    use_8bit = overrides.get("use_8bit") if "use_8bit" in overrides else base.use_8bit
    show_progress = overrides.get("show_progress") if "show_progress" in overrides else base.show_progress

    resolved_template = _resolve_prompt_template(prompt_template, default_template)

    # Use registry settings - quantization specified there
    if use_8bit is None:
        use_8bit = True  # Default to 8-bit quantization (standard for this project)

    # Create comparison object
    comparison = ModelComparison(
        category=category,
        test_data_path=test_data_path,
        prompt_template=resolved_template,
        max_samples=max_samples,
        use_8bit=bool(use_8bit),
        show_progress=bool(show_progress),
    )

    results = comparison.compare_all_models()
    comparison.print_summary()

    # All results automatically saved to results/experiment_history/ by ExperimentTracker
    print(f"\n✓ All experiments saved to results/experiment_history/{category}.json")

    return results
