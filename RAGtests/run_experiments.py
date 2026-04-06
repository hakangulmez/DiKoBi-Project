"""
Few-shot experiment runner.
Tests different combinations of few-shot strategies with MSE focus.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RAGtests.config import *
from RAGtests.few_shot_selector import FewShotSelector
from RAGtests.prompts.few_shot_templates import FewShotPromptGenerator
from RAGtests.mse_evaluator import MSEEvaluator, compare_multiple_results
from src.classification.classifier import TextClassifier
from src.utils.device import get_device, print_device_info


class FewShotExperimentRunner:
    """Runs few-shot prompting experiments with MSE optimization."""
    
    def __init__(
        self,
        category: str,
        model_name: str = DEFAULT_MODEL,
        use_8bit: bool = True,
        test_size: Optional[int] = DEFAULT_TEST_SIZE,
        random_seed: int = RANDOM_SEED
    ):
        """
        Initialize experiment runner.
        
        Args:
            category: Category to test (e.g., "1_D_M")
            model_name: Model to use
            use_8bit: Use 8-bit quantization
            test_size: Number of test samples (None = full test set)
            random_seed: Random seed for reproducibility
        """
        self.category = category
        self.model_name = model_name
        self.use_8bit = use_8bit
        self.test_size = test_size
        self.random_seed = random_seed
        
        # Set up paths
        self.train_path = DATA_DIR / "train" / f"{category}.csv"
        self.test_path = DATA_DIR / "test" / f"{category}.csv"
        
        # Verify data exists
        if not self.train_path.exists():
            raise FileNotFoundError(f"Training data not found: {self.train_path}")
        if not self.test_path.exists():
            raise FileNotFoundError(f"Test data not found: {self.test_path}")
        
        # Load test data
        print(f"Loading test data from: {self.test_path}")
        self.test_df = pd.read_csv(self.test_path)
        
        # Limit test size if specified
        if test_size is not None:
            np.random.seed(random_seed)
            self.test_df = self.test_df.sample(n=min(test_size, len(self.test_df)))
        
        print(f"✓ Test data loaded: {len(self.test_df)} samples")
        
        # Initialize components
        self.selector = FewShotSelector(self.train_path, category, random_seed)
        self.prompt_generator = FewShotPromptGenerator(category)
        self.classifier = None
        
        # Results storage
        self.results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_model(self):
        """Load the classification model."""
        if self.classifier is None:
            print(f"\n🔧 Loading model: {self.model_name}")
            print_device_info()
            
            self.classifier = TextClassifier(
                model_name=self.model_name,
                use_8bit=self.use_8bit,
                batch_size=BATCH_SIZE
            )
            print("✓ Model loaded successfully")
    
    def run_single_experiment(
        self,
        n_shots: int,
        selection_strategy: str,
        prompt_format: str,
        experiment_name: Optional[str] = None
    ) -> Dict:
        """
        Run a single few-shot experiment.
        
        Args:
            n_shots: Number of examples to use
            selection_strategy: Example selection strategy
            prompt_format: Prompt format type
            experiment_name: Optional custom name
            
        Returns:
            Dictionary with results
        """
        if experiment_name is None:
            experiment_name = f"{n_shots}shot_{selection_strategy}_{prompt_format}"
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {experiment_name}")
        print(f"{'='*70}")
        print(f"  N-shots: {n_shots}")
        print(f"  Selection: {selection_strategy}")
        print(f"  Format: {prompt_format}")
        
        # Select examples
        print(f"\n📝 Selecting {n_shots} examples using '{selection_strategy}' strategy...")
        examples = self.selector.select_examples(n_shots, selection_strategy)
        stats = self.selector.get_example_statistics(examples)
        
        print(f"  ✓ Selected {stats['n_examples']} examples")
        print(f"  Rating distribution: {stats['rating_distribution']}")
        print(f"  Avg text length: {stats['avg_text_length']:.0f} chars")
        
        # Generate prompts
        print(f"\n🔮 Generating prompts with '{prompt_format}' format...")
        prompts = []
        for _, row in self.test_df.iterrows():
            prompt = self.prompt_generator.generate_prompt(
                row['free_text_answer'],
                examples,
                prompt_format
            )
            prompts.append(prompt)
        
        print(f"  ✓ Generated {len(prompts)} prompts")
        print(f"  Avg prompt length: {np.mean([len(p) for p in prompts]):.0f} chars")
        
        # Run classification
        print(f"\n🤖 Running classification...")
        start_time = time.time()
        
        self.load_model()
        
        # Get predictions (using custom prompts)
        predictions = []
        failed_indices = []
        
        for i, prompt in enumerate(prompts):
            try:
                # Use classifier's model directly with custom prompt
                inputs = self.classifier.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.classifier.device)
                
                outputs = self.classifier.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.classifier.tokenizer.pad_token_id
                )
                
                response = self.classifier.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                
                # Extract score
                score = self.classifier._extract_score(
                    response,
                    sorted(self.selector.classes)
                )
                
                if score is not None:
                    predictions.append(score)
                else:
                    predictions.append(None)
                    failed_indices.append(i)
                    
            except Exception as e:
                print(f"  ⚠️  Error on sample {i}: {e}")
                predictions.append(None)
                failed_indices.append(i)
        
        elapsed_time = time.time() - start_time
        
        # Remove failed predictions
        valid_mask = [p is not None for p in predictions]
        y_true = self.test_df[self.category].values[valid_mask]
        y_pred = np.array([p for p in predictions if p is not None])
        
        n_failed = len(failed_indices)
        n_processed = len(y_pred)
        
        print(f"  ✓ Classification complete")
        print(f"  Processed: {n_processed}/{len(prompts)} samples")
        print(f"  Failed: {n_failed}")
        print(f"  Time: {elapsed_time:.1f}s ({elapsed_time/len(prompts):.2f}s per sample)")
        
        # Evaluate
        print(f"\n📊 Evaluating results...")
        evaluator = MSEEvaluator(y_true, y_pred)
        metrics = evaluator.calculate_all_metrics()
        
        # Print summary
        evaluator.print_summary(experiment_name)
        
        # Store results
        result = {
            'experiment_name': experiment_name,
            'category': self.category,
            'model': self.model_name,
            'n_shots': n_shots,
            'selection_strategy': selection_strategy,
            'prompt_format': prompt_format,
            'example_stats': stats,
            'metrics': metrics,
            'n_processed': n_processed,
            'n_failed': n_failed,
            'time_seconds': elapsed_time,
            'time_per_sample': elapsed_time / len(prompts),
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(result)
        
        return result
    
    def run_strategy_comparison(
        self,
        n_shots: int = 3,
        strategies: Optional[List[str]] = None,
        prompt_format: str = "standard"
    ):
        """
        Compare different example selection strategies.
        
        Args:
            n_shots: Number of examples to use
            strategies: List of strategies to test (None = all)
            prompt_format: Prompt format to use
        """
        if strategies is None:
            strategies = SELECTION_STRATEGIES
        
        print(f"\n{'#'*70}")
        print(f"STRATEGY COMPARISON: Testing {len(strategies)} strategies")
        print(f"{'#'*70}")
        
        for strategy in strategies:
            try:
                self.run_single_experiment(
                    n_shots=n_shots,
                    selection_strategy=strategy,
                    prompt_format=prompt_format
                )
            except Exception as e:
                print(f"\n❌ Error in strategy '{strategy}': {e}")
                continue
        
        # Print comparison
        self._print_comparison("Strategy Comparison")
    
    def run_format_comparison(
        self,
        n_shots: int = 3,
        selection_strategy: str = "balanced",
        formats: Optional[List[str]] = None
    ):
        """
        Compare different prompt formats.
        
        Args:
            n_shots: Number of examples to use
            selection_strategy: Example selection strategy
            formats: List of formats to test (None = all)
        """
        if formats is None:
            formats = PROMPT_FORMATS
        
        print(f"\n{'#'*70}")
        print(f"FORMAT COMPARISON: Testing {len(formats)} formats")
        print(f"{'#'*70}")
        
        for format_type in formats:
            try:
                self.run_single_experiment(
                    n_shots=n_shots,
                    selection_strategy=selection_strategy,
                    prompt_format=format_type
                )
            except Exception as e:
                print(f"\n❌ Error in format '{format_type}': {e}")
                continue
        
        # Print comparison
        self._print_comparison("Format Comparison")
    
    def run_n_shots_comparison(
        self,
        n_shots_list: Optional[List[int]] = None,
        selection_strategy: str = "balanced",
        prompt_format: str = "standard"
    ):
        """
        Compare different numbers of examples.
        
        Args:
            n_shots_list: List of n_shots to test (None = default)
            selection_strategy: Example selection strategy
            prompt_format: Prompt format to use
        """
        if n_shots_list is None:
            n_shots_list = DEFAULT_NUM_SHOTS
        
        print(f"\n{'#'*70}")
        print(f"N-SHOTS COMPARISON: Testing {n_shots_list}")
        print(f"{'#'*70}")
        
        for n_shots in n_shots_list:
            try:
                self.run_single_experiment(
                    n_shots=n_shots,
                    selection_strategy=selection_strategy,
                    prompt_format=prompt_format
                )
            except Exception as e:
                print(f"\n❌ Error with {n_shots} shots: {e}")
                continue
        
        # Print comparison
        self._print_comparison("N-Shots Comparison")
    
    def run_full_experiment_suite(self):
        """Run comprehensive experiment suite."""
        print(f"\n{'#'*70}")
        print(f"FULL EXPERIMENT SUITE")
        print(f"Category: {self.category}")
        print(f"Model: {self.model_name}")
        print(f"Test samples: {len(self.test_df)}")
        print(f"{'#'*70}")
        
        # 1. N-shots comparison (balanced strategy, standard format)
        print(f"\n\n### PHASE 1: N-Shots Comparison ###")
        self.run_n_shots_comparison(
            n_shots_list=[1, 3, 5, 7],
            selection_strategy="balanced",
            prompt_format="standard"
        )
        
        # 2. Strategy comparison (3-shot, standard format)
        print(f"\n\n### PHASE 2: Strategy Comparison ###")
        self.run_strategy_comparison(
            n_shots=3,
            strategies=["random", "balanced", "stratified", "diverse", "representative"],
            prompt_format="standard"
        )
        
        # 3. Format comparison (3-shot, balanced strategy)
        print(f"\n\n### PHASE 3: Format Comparison ###")
        self.run_format_comparison(
            n_shots=3,
            selection_strategy="balanced",
            formats=PROMPT_FORMATS
        )
        
        # Final summary
        self._print_final_summary()
    
    def _print_comparison(self, title: str):
        """Print comparison table for recent results."""
        if not self.results:
            return
        
        print(f"\n{'='*70}")
        print(f"{title.upper()}")
        print(f"{'='*70}")
        
        # Extract results for comparison
        result_data = {}
        for r in self.results:
            result_data[r['experiment_name']] = (
                r['metrics']['confusion_matrix'],  # Placeholder, we'll extract properly
                r['metrics']
            )
        
        # Create comparison dataframe
        comparison = []
        for r in self.results:
            comparison.append({
                'Experiment': r['experiment_name'],
                'MSE': r['metrics']['mse'],
                'RMSE': r['metrics']['rmse'],
                'MAE': r['metrics']['mae'],
                'Accuracy': r['metrics']['accuracy'],
                'Time(s)': r['time_seconds']
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('MSE')
        df.insert(0, 'Rank', range(1, len(df) + 1))
        
        print(df.to_string(index=False))
        
        # Highlight best
        best = df.iloc[0]
        print(f"\n⭐ BEST: {best['Experiment']}")
        print(f"   MSE: {best['MSE']:.4f} | Accuracy: {best['Accuracy']:.1%}")
        print(f"{'='*70}\n")
    
    def _print_final_summary(self):
        """Print final summary of all experiments."""
        print(f"\n{'#'*70}")
        print(f"FINAL SUMMARY - ALL EXPERIMENTS")
        print(f"{'#'*70}")
        
        # Create full comparison
        comparison = []
        for r in self.results:
            comparison.append({
                'Experiment': r['experiment_name'],
                'N-Shots': r['n_shots'],
                'Strategy': r['selection_strategy'],
                'Format': r['prompt_format'],
                'MSE': r['metrics']['mse'],
                'Accuracy': r['metrics']['accuracy'],
                'Time(s)': r['time_seconds']
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('MSE')
        df.insert(0, 'Rank', range(1, len(df) + 1))
        
        print(df.to_string(index=False))
        
        # Overall best
        best = df.iloc[0]
        print(f"\n{'='*70}")
        print(f"🏆 OVERALL BEST CONFIGURATION:")
        print(f"{'='*70}")
        print(f"  N-Shots:  {best['N-Shots']}")
        print(f"  Strategy: {best['Strategy']}")
        print(f"  Format:   {best['Format']}")
        print(f"  MSE:      {best['MSE']:.4f} ⭐")
        print(f"  Accuracy: {best['Accuracy']:.1%}")
        print(f"  Time:     {best['Time(s)']:.1f}s")
        print(f"{'='*70}\n")
    
    def save_results(self, output_dir: Optional[Path] = None):
        """Save results to JSON file."""
        if output_dir is None:
            output_dir = RESULTS_DIR
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"few_shot_results_{self.category}_{self.experiment_id}.json"
        output_path = output_dir / filename
        
        # Prepare data for JSON
        output_data = {
            'experiment_id': self.experiment_id,
            'category': self.category,
            'model': self.model_name,
            'test_size': self.test_size,
            'random_seed': self.random_seed,
            'timestamp': datetime.now().isoformat(),
            'results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.classifier is not None:
            print("\n🧹 Cleaning up...")
            self.classifier.unload()
            self.classifier = None


def main():
    """Example usage."""
    # Test configuration
    category = "1_D_M"
    model = "Qwen/Qwen2.5-3B-Instruct"
    test_size = 50  # Small test for quick iteration
    
    # Create runner
    runner = FewShotExperimentRunner(
        category=category,
        model_name=model,
        use_8bit=True,
        test_size=test_size
    )
    
    try:
        # Run full experiment suite
        runner.run_full_experiment_suite()
        
        # Save results
        runner.save_results()
        
    finally:
        # Cleanup
        runner.cleanup()


if __name__ == "__main__":
    main()
