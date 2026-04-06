"""
Prompt refinement testing tool.
Compare different prompt variations to find the best one.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

from ..classification import TextClassifier
from ..classification.prompts import list_prompt_templates, get_prompt
from .metrics import calculate_metrics, compare_metrics
from ..utils import ExperimentTracker
from ..models import get_model_config

logger = logging.getLogger(__name__)


class PromptTester:
    """
    Test different prompt variations to find the best one.
    
    This is Phase 2 of the workflow: prompt refinement before model comparison.
    Use a small, fast model to quickly iterate on prompts.
    
    Example:
        tester = PromptTester(
            category="1_D_M",
            test_data_path="data/processed/categories/1_D_M.csv",
            model_name="Qwen/Qwen2.5-0.5B-Instruct",  # Fast model
            max_samples=20  # Small sample for quick iteration
        )
        
        results = tester.test_all_prompts()
        tester.save_report("results/prompt_testing/")
    """
    
    def __init__(
        self,
        category: str,
        test_data_path: str,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_samples: Optional[int] = 20,
        use_8bit: bool = False,
    ):
        """
        Initialize prompt tester.
        
        Args:
            category: DiKoBi category to test
            test_data_path: Path to CSV with test data
            model_name: Model to use (recommend small/fast)
            max_samples: Limit samples for quick testing
            use_8bit: Use 8-bit quantization
        """
        self.category = category
        self.test_data_path = test_data_path
        self.model_name = model_name
        self.max_samples = max_samples
        
        # Resolve model_name to HF ID for consistent experiment tracking
        try:
            model_config = get_model_config(model_name)
            self.model_hf_id = model_config.hf_id
        except:
            # If not in registry, assume it's already an HF ID
            self.model_hf_id = model_name
        
        # Load test data
        logger.info(f"Loading test data from {test_data_path}")
        df_full = pd.read_csv(test_data_path)
        self.total_samples = int(len(df_full))
        self.df = df_full
        
        if max_samples:
            self.df = self.df.head(max_samples)
            logger.info(f"Limited to {max_samples} samples for quick testing")
        
        logger.info(f"Test set size: {len(self.df)}")
        
        # Initialize classifier
        logger.info(f"Loading model: {model_name}")
        self.classifier = TextClassifier(
            model_name=model_name,
            use_8bit=use_8bit,
            batch_size=2,
        )
        
        # Store output format from classifier
        self.output_format = "json" if self.classifier.use_json else "text"
        
        # Results storage
        self.results: Dict[str, Dict] = {}
        
        # Experiment tracker
        self.tracker = ExperimentTracker(category)
        self.use_8bit = use_8bit
    
    def test_single_prompt(
        self,
        prompt_template: str,
        prompt_name: str,
        show_raw: bool = False,
        return_predictions: bool = False,
    ) -> Dict:
        """
        Test a single prompt template.
        
        Note: This method assumes the caller has already checked for duplicates.
        Use compare_all_templates() or test_custom_prompt() which handle duplicate
        checking before loading the model.
        
        Args:
            prompt_template: Template name to test
            prompt_name: Display name for results
            show_raw: Show raw model outputs and parsing statistics
        
        Returns:
            Dictionary with results and metrics
        """
        logger.info(f"Testing prompt: {prompt_name}")
        
        # Get data and run predictions
        texts = self.df['free_text_answer'].tolist()
        true_labels = self.df[self.category].tolist()
        
        # Run predictions with raw outputs for failure inspection
        results = self.classifier.classify_batch(
            category=self.category,
            texts=texts,
            prompt_template=prompt_template,
            show_progress=True,  # Show progress bar
            return_raw=show_raw  # Only get raw if debugging
        )
        
        # Extract predictions and collect failures
        y_pred = [r['score'] for r in results if r['score'] is not None]
        y_true = [true_labels[i] for i, r in enumerate(results) if r['score'] is not None]

        predictions = []
        if return_predictions:
            for i, r in enumerate(results):
                y_pred_i = r.get("score")
                predictions.append(
                    {
                        "sample_index": i,
                        "text": texts[i],
                        "y_true": true_labels[i],
                        "y_pred": y_pred_i,
                        "parse_ok": y_pred_i is not None,
                        "raw_output": r.get("raw_output", "") if show_raw else "",
                    }
                )
        
        # Collect failed parses if show_raw enabled
        failed_parses = []
        if show_raw:
            from ..classification.prompts import get_valid_outputs
            valid_outputs = get_valid_outputs(self.category)
            failed_parses = [
                {
                    "index": i,
                    "text": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                    "raw_output": r.get('raw_output', 'N/A')[:200],
                    "expected_range": valid_outputs
                }
                for i, r in enumerate(results)
                if r['score'] is None and 'raw_output' in r
            ]
        
        # Calculate metrics
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = calculate_metrics(y_true, y_pred)
        
        samples_skipped = len(texts) - len(y_pred)
        logger.info(f"  Accuracy: {metrics['accuracy']:.2%}, QWK: {metrics['qwk']:.4f}")
        if samples_skipped > 0:
            logger.info(f"  Skipped: {samples_skipped}/{len(texts)} failed to parse")
        
        result_dict = {
            "prompt_name": prompt_name,
            "prompt_template": prompt_template,
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": y_pred,
            "samples_processed": len(y_pred),
            "samples_skipped": samples_skipped
        }

        if return_predictions:
            result_dict["predictions"] = predictions
        
        # Show examples if debugging
        if show_raw and failed_parses:
            result_dict["failed_parses"] = failed_parses
            self._print_parsing_failures(failed_parses, samples_skipped, len(texts))
        
        return result_dict
    
    def _print_parsing_failures(self, failures: List[Dict], total_failed: int, total_samples: int):
        """Print parsing failure examples."""
        print("\n" + "="*80)
        print("PARSING FAILURES")
        print("="*80)
        print(f"Total failed: {total_failed}/{total_samples} ({total_failed/total_samples:.1%})")
        
        if failures:
            print(f"\nShowing first {len(failures)} examples:\n")
            for i, failure in enumerate(failures, 1):
                print(f"[{i}] Sample {failure['index']}")
                print(f"    Input: {failure['text']}")
                print(f"    Output: {failure['raw_output']}")
                print(f"    Expected: one of {failure['expected_range']}")
                print()
        
        print("💡 RECOMMENDATION:")
        if total_failed / total_samples > 0.1:
            print("  ⚠️  >10% failures - consider refining prompts or parser")
        else:
            print("  ✓ Low failure rate - parser working well")
        print("="*80 + "\n")
    
    def test_all_templates(
        self, 
        custom_prompts: Optional[Dict[str, str]] = None,
        show_raw: bool = False
    ) -> Dict:
        """
        Test all available prompt templates.
        
        Args:
            custom_prompts: Optional dict of {name: template_key} to test
            show_raw: Show raw model outputs and parsing statistics
        
        Returns:
            Dictionary mapping prompt names to results
        """
        # Get available prompts
        if custom_prompts:
            prompts_to_test = custom_prompts
        else:
            # Use all built-in templates
            prompts_to_test = list_prompt_templates()
        
        logger.info(f"Testing {len(prompts_to_test)} prompt variations")
        
        for idx, (name, template_key) in enumerate(prompts_to_test.items(), 1):
            print(f"\n[{idx}/{len(prompts_to_test)}] Testing: {name}")
            try:
                result = self.test_single_prompt(
                    template_key,
                    name,
                    show_raw=show_raw,
                    return_predictions=True,
                )

                predictions = result.pop("predictions", None)
                self.results[name] = result
                
                # Clear cache after each test to prevent memory buildup
                self.classifier.clear_cache()

                # Log to experiment tracker
                sample_text = "[SAMPLE TEXT]"
                prompt_text = get_prompt(
                    self.category,
                    sample_text,
                    template_name=template_key,
                    output_format=self.output_format,
                )
                quantization = "8bit" if self.use_8bit else "none"
                self.tracker.log_experiment(
                    model=self.model_hf_id,
                    prompt_template=template_key,
                    prompt_text=prompt_text,
                    metrics=result["metrics"],
                    dataset_path=self.test_data_path,
                    total_samples=self.total_samples,
                    samples_used=result["samples_processed"],
                    max_samples=self.max_samples,
                    quantization=quantization,
                    output_format=self.output_format,
                    generation_params=self.classifier.get_generation_params(),
                    version_name=name,
                    predictions=predictions,
                )
                
                # Show basic stats
                skipped = result.get('samples_skipped', 0)
                skip_str = f", {skipped} failed" if skipped > 0 else ""
                print(f"✓ Completed {name}: Accuracy={result['metrics']['accuracy']:.1%}, "
                      f"QWK={result['metrics']['qwk']:.3f}{skip_str}")
            except Exception as e:
                logger.error(f"Failed to test prompt '{name}': {e}")
                self.results[name] = {"error": str(e)}
                print(f"✗ Failed {name}: {e}")
        
        return self.results
    
    def print_summary(self):
        """Print summary of all tested prompts."""
        if not self.results:
            print("No results to display. Run test_all_prompts() first.")
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
        
        print("\n" + "=" * 80)
        print("PROMPT TESTING RESULTS")
        print("=" * 80)
        print(f"Category: {self.category}")
        print(f"Model: {self.model_name}")
        print(f"Samples: {self.max_samples}")
        print("\n" + "-" * 90)
        print(f"{'Rank':<6} {'Prompt':<25} {'Accuracy':<12} {'QWK':<12} {'MSE':<10} {'Failed':<8}")
        print("-" * 90)
        
        for rank, (name, result) in enumerate(sorted_results, 1):
            m = result['metrics']
            skipped = result.get('samples_skipped', 0)
            marker = "⭐" if rank == 1 else "  "
            skip_str = f"{skipped}" if skipped > 0 else "-"
            print(f"{marker} {rank:<4} {name:<25} {m['accuracy']:<12.2%} "
                  f"{m['qwk']:<12.4f} {m['mse']:<10.4f} {skip_str:<8}")
        
        print("=" * 80)
        
        # Print recommendation
        best_name = sorted_results[0][0]
        best_qwk = sorted_results[0][1]['metrics']['qwk']
        print(f"\n✓ RECOMMENDED: '{best_name}' (QWK: {best_qwk:.4f})")
        print()
    
    def save_report(self, output_dir: str):
        """
        Save detailed report to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save text report
        report_path = output_path / f"prompt_test_{self.category}_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PROMPT TESTING REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Category: {self.category}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Test Samples: {self.max_samples}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Sort by QWK
            valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
            sorted_results = sorted(
                valid_results.items(),
                key=lambda x: x[1]['metrics']['qwk'],
                reverse=True
            )
            
            f.write("RANKINGS (by QWK):\n")
            f.write("-" * 80 + "\n")
            
            for rank, (name, result) in enumerate(sorted_results, 1):
                f.write(f"\n{rank}. {name}\n")
                f.write(f"   Accuracy: {result['metrics']['accuracy']:.2%}\n")
                f.write(f"   QWK:      {result['metrics']['qwk']:.4f}\n")
                f.write(f"   MSE:      {result['metrics']['mse']:.4f}\n")
                f.write(f"   MAE:      {result['metrics']['mae']:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"RECOMMENDATION: Use '{sorted_results[0][0]}'\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Report saved to {report_path}")
        
        # Save CSV
        csv_path = output_path / f"prompt_test_{self.category}_{timestamp}.csv"
        
        rows = []
        for name, result in self.results.items():
            if 'error' not in result:
                rows.append({
                    'prompt_name': name,
                    'accuracy': result['metrics']['accuracy'],
                    'qwk': result['metrics']['qwk'],
                    'mse': result['metrics']['mse'],
                    'mae': result['metrics']['mae'],
                    'samples_processed': result['samples_processed']
                })
        
        df_results = pd.DataFrame(rows)
        df_results.to_csv(csv_path, index=False)
        logger.info(f"CSV saved to {csv_path}")
        
        # Also save to unified experiment tracker
        self.save_to_tracker()
    
    def save_to_tracker(self):
        """Save all results to unified experiment tracker."""
        for name, result in self.results.items():
            if 'error' in result:
                continue
            
            # Get full prompt text
            sample_text = "[SAMPLE TEXT]"
            prompt_text = get_prompt(
                self.category,
                sample_text,
                template_name=result['prompt_template'],
                output_format=self.output_format
            )
            
            # Log to tracker
            quantization = "8bit" if self.use_8bit else "none"
            self.tracker.log_experiment(
                model=self.model_hf_id,
                prompt_template=result['prompt_template'],
                prompt_text=prompt_text,
                metrics=result['metrics'],
                dataset_path=self.test_data_path,
                total_samples=self.total_samples,
                samples_used=result['samples_processed'],
                max_samples=self.max_samples,
                quantization=quantization,
                output_format=self.output_format,
                generation_params=self.classifier.get_generation_params(),
                version_name=name
            )
        
        logger.info(f"Results saved to unified experiment tracker")
    
    def cleanup(self):
        """Free resources and clear memory."""
        import gc
        import torch
        
        self.classifier.unload()
        
        # Additional cleanup
        gc.collect()
        
        # Platform-specific cache clearing
        from ..utils.device import get_device
        device = get_device()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
            gc.collect()  # Extra pass for MPS


def compare_all_templates(
    category: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    max_samples: Optional[int] = None,
    use_8bit: bool = True,
    show_progress: bool = True,
) -> Dict[str, Dict]:
    """
    Run template comparison (Phase 1).
    
    Automatically tests different prompting approaches (zero-shot, few-shot)
    to find the best baseline before manual prompt development.
    
    Args:
        category: Category to test (e.g., "1_D_M")
        model_name: Model to use for testing
        max_samples: Maximum samples to test (None = full test set)
        use_8bit: Use 8-bit quantization
        show_progress: Show progress bars
    
    Returns:
        Dictionary with results for each template tested
    
    Example:
        from src.evaluation import compare_all_templates
        
        results = compare_all_templates(
            category="1_D_M",
            model_name="Qwen/Qwen2.5-7B-Instruct",
            max_samples=None,
            use_8bit=True,
        )
    """
    from pathlib import Path
    import json
    from ..models import MODELS, get_model_config
    
    test_data_path = f"data/processed/test/{category}.csv"
    
    # Check if data exists
    if not Path(test_data_path).exists():
        raise FileNotFoundError(f"Test data not found at {test_data_path}")

    # Get templates to test
    templates = list_prompt_templates()

    # CHECK FOR DUPLICATES BEFORE LOADING MODEL
    # Create tracker and check which templates need to be run
    tracker = ExperimentTracker(category)
    
    # Check which templates are already cached
    templates_to_run = []
    cached_results = {}
    
    print("\nChecking for cached experiments...")
    for template_name in templates:
        result = tracker.check_duplicate(
            model=model_name,
            prompt_template=template_name,
            max_samples=max_samples,
            use_8bit=use_8bit,
            verbose=True,
        )
        
        cached = result['cached']
        if cached:
            # Store cached result
            metrics = cached.get('metrics', {})
            cached_results[template_name] = {
                "prompt_name": template_name,
                "prompt_template": template_name,
                "metrics": metrics,
                "samples_processed": cached['dataset']['samples_used'],
                "samples_skipped": 0,
                "from_cache": True,
                "experiment_id": cached.get('experiment_id')
            }
            print(f"  ✓ {template_name}: Using cached results")
        else:
            templates_to_run.append(template_name)
            print(f"  • {template_name}: Needs to run")
    
    # If all templates are cached, return results without loading model
    if not templates_to_run:
        print("\n✓ All templates already cached. Skipping model loading.")
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY (from cache)")
        print("=" * 80)
        for name, result in cached_results.items():
            print(f"\n{name}:")
            print(f"  Accuracy: {result['metrics']['accuracy']:.1%}")
            print(f"  QWK: {result['metrics']['qwk']:.3f}")
        
        # Show recommendation
        best_template = max(cached_results.items(), key=lambda x: x[1]['metrics']['qwk'])
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        print(f"\n✅ Best template: {best_template[0]}")
        print(f"   QWK: {best_template[1]['metrics']['qwk']:.3f}")
        print(f"   Accuracy: {best_template[1]['metrics']['accuracy']:.3f}")
        print(f"\n💡 To use this in Phase 2/3:")
        print(f"   Set TEMPLATE = \"{best_template[0]}\" in the configuration cell")
        print("=" * 80 + "\n")
        
        return cached_results
    
    # Some templates need to run - initialize tester with model
    print(f"\nLoading model for {len(templates_to_run)} template(s)...")
    tester = PromptTester(
        category=category,
        test_data_path=test_data_path,
        model_name=model_name,
        max_samples=max_samples,
        use_8bit=use_8bit,
    )
    
    # Display full configuration
    output_format = "JSON" if tester.classifier.use_json else "Text"
    print("\n" + "=" * 80)
    print("PHASE 1: TEMPLATE COMPARISON (single category)")
    print("=" * 80)
    print(f"\n  Category:     {category}")
    print(f"  Model (key):  {model_name}")
    print(f"  HF ID:        {tester.model_hf_id}")
    print(f"  Output:       {output_format}")
    print(f"  Max Tokens:   {tester.classifier.max_new_tokens}")
    print(f"  Samples:      {max_samples or 'Full test set'}")
    print(f"  Quantization: {'8-bit' if use_8bit else 'None'}")
    print(f"  Templates:    {len(templates_to_run)} to run, {len(cached_results)} cached")
    print("=" * 80)
    
    print(f"\nTesting {len(templates_to_run)} template(s): {', '.join(templates_to_run)}")
    print("\nThis may take several minutes...\n")
    
    # Test only the templates that need to run
    custom_prompts = {name: name for name in templates_to_run}
    new_results = tester.test_all_templates(custom_prompts=custom_prompts, show_raw=False)
    
    # Merge cached and new results
    all_results = {**cached_results, **new_results}
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Print all results (cached and new)
    for name in templates:
        if name in all_results and 'error' not in all_results[name]:
            result = all_results[name]
            cache_str = " (cached)" if result.get('from_cache', False) else ""
            print(f"\n{name}{cache_str}:")
            print(f"  Accuracy: {result['metrics']['accuracy']:.1%}")
            print(f"  QWK: {result['metrics']['qwk']:.3f}")
    
    # All results automatically saved to results/experiment_history/ by ExperimentTracker
    print(f"\n✓ All experiments saved to results/experiment_history/{category}.json")
    print(f"✓ Per-run artifacts saved to data/experiments/ (predictions CSV + experiment JSON)")
    
    # Show recommendation
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    if valid_results:
        best_template = max(
            valid_results.items(),
            key=lambda x: x[1]['metrics']['qwk']
        )
        
        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        print(f"\n✅ Best template: {best_template[0]}")
        print(f"   QWK: {best_template[1]['metrics']['qwk']:.3f}")
        print(f"   Accuracy: {best_template[1]['metrics']['accuracy']:.3f}")
        print(f"\n💡 To use this in Phase 2/3:")
        print(f"   Set TEMPLATE = \"{best_template[0]}\" in the configuration cell")
        print("=" * 80 + "\n")
    
    # Cleanup
    tester.cleanup()
    
    return all_results


def test_custom_prompt(
    category: str,
    template: str,
    custom_instructions: str,
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    max_samples: Optional[int] = 20,
    use_8bit: bool = False,
    show_progress: bool = True,
) -> Dict:
    """
    Test a custom prompt variation.
    
    This is for Phase 2: test your custom prompt against the baseline.
    
    Args:
        category: Category to test (e.g., "1_D_M")
        template: Base template ("zero_shot" or "few_shot")
        custom_instructions: Your custom task instructions
        model_name: Model to use (fast model recommended)
        max_samples: Number of samples to test (None = all)
        use_8bit: Use 8-bit quantization
        show_progress: Show progress bars
    
    Returns:
        Dictionary with results and metrics
    
    Example:
        result = test_custom_prompt(
            category="1_D_M",
            template="few_shot",
            custom_instructions="Your custom prompt here",
            max_samples=20
        )
    """
    from pathlib import Path
    import json
    from ..models import get_model_config
    
    test_data_path = Path("data/processed/test") / f"{category}.csv"
    
    # CHECK FOR DUPLICATES BEFORE LOADING MODEL
    tracker = ExperimentTracker(category)
    
    check_result = tracker.check_duplicate(
        model=model_name,
        prompt_template=template,
        max_samples=max_samples,
        use_8bit=use_8bit,
        verbose=True,
    )
    
    cached = check_result['cached']
    if cached:
        # Return cached results without loading model
        logger.info("✓ Using cached experiment results (skipped model loading)")
        metrics = cached.get('metrics', {})
        return {
            "prompt_name": f"custom_{template}",
            "prompt_template": template,
            "metrics": metrics,
            "samples_processed": cached['dataset']['samples_used'],
            "samples_skipped": 0,
            "from_cache": True,
            "experiment_id": cached.get('experiment_id')
        }
    
    # NO DUPLICATE - Initialize tester and run predictions
    logger.info("No cached results found. Loading model and running predictions...")
    tester = PromptTester(
        category=category,
        test_data_path=str(test_data_path),
        model_name=model_name,
        max_samples=max_samples,
        use_8bit=use_8bit,
    )
    
    # Test custom prompt
    result = tester.test_single_prompt(
        prompt_template=template,
        prompt_name=f"custom_{template}",
        show_raw=False,
        return_predictions=True,
    )

    predictions = result.pop("predictions", None)

    # Log this custom run as its own experiment (+ predictions CSV)
    # Use metadata computed by check_duplicate
    tester.tracker.log_experiment(
        model=check_result['model_hf_id'],
        prompt_template=template,
        prompt_text=check_result['prompt_text'],
        metrics=result["metrics"],
        dataset_path=check_result['dataset_path'],
        total_samples=tester.total_samples,
        samples_used=result["samples_processed"],
        max_samples=max_samples,
        quantization=check_result['quantization'],
        output_format=check_result['output_format'],
        generation_params=tester.classifier.get_generation_params(),
        version_name=f"custom_{template}",
        notes=custom_instructions,
        predictions=predictions,
    )
    
    # Cleanup
    tester.cleanup()
    
    return result
