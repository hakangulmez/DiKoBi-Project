"""
Batch runner for testing multiple categories.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RAGtests.run_experiments import FewShotExperimentRunner
from RAGtests.config import QUICK_TEST_CATEGORIES, ALL_CATEGORIES, DEFAULT_MODEL


def run_batch_experiments(
    categories=None,
    test_size=50,
    phases=None
):
    """
    Run experiments on multiple categories.
    
    Args:
        categories: List of categories (None = quick test categories)
        test_size: Number of test samples per category
        phases: List of phases to run (None = all phases)
                Options: ["n_shots", "strategies", "formats", "full"]
    """
    if categories is None:
        categories = QUICK_TEST_CATEGORIES
    
    if phases is None:
        phases = ["full"]  # Default to full suite
    
    print(f"\n{'#'*70}")
    print(f"BATCH EXPERIMENT RUNNER")
    print(f"{'#'*70}")
    print(f"Categories: {len(categories)}")
    print(f"Test size per category: {test_size}")
    print(f"Phases: {phases}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")
    
    all_results = {}
    
    for i, category in enumerate(categories, 1):
        print(f"\n{'='*70}")
        print(f"CATEGORY {i}/{len(categories)}: {category}")
        print(f"{'='*70}\n")
        
        runner = FewShotExperimentRunner(
            category=category,
            model_name=DEFAULT_MODEL,
            use_8bit=True,
            test_size=test_size
        )
        
        try:
            if "full" in phases:
                # Run full experiment suite
                runner.run_full_experiment_suite()
            else:
                # Run selected phases
                if "n_shots" in phases:
                    runner.run_n_shots_comparison()
                
                if "strategies" in phases:
                    runner.run_strategy_comparison(n_shots=3)
                
                if "formats" in phases:
                    runner.run_format_comparison(n_shots=3)
            
            # Save results
            runner.save_results()
            
            # Store summary
            if runner.results:
                best_result = min(runner.results, key=lambda x: x['metrics']['mse'])
                all_results[category] = {
                    'best_mse': best_result['metrics']['mse'],
                    'best_accuracy': best_result['metrics']['accuracy'],
                    'best_config': {
                        'n_shots': best_result['n_shots'],
                        'strategy': best_result['selection_strategy'],
                        'format': best_result['prompt_format']
                    }
                }
        
        except Exception as e:
            print(f"\n❌ Error processing category {category}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        finally:
            runner.cleanup()
    
    # Print overall summary
    print(f"\n{'#'*70}")
    print(f"BATCH SUMMARY")
    print(f"{'#'*70}")
    print(f"Completed: {len(all_results)}/{len(categories)} categories")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if all_results:
        print(f"\n{'='*70}")
        print("BEST RESULTS PER CATEGORY")
        print(f"{'='*70}")
        
        for category, result in all_results.items():
            print(f"\n{category}:")
            print(f"  MSE: {result['best_mse']:.4f}")
            print(f"  Accuracy: {result['best_accuracy']:.1%}")
            print(f"  Config: {result['best_config']['n_shots']}-shot, "
                  f"{result['best_config']['strategy']}, {result['best_config']['format']}")
    
    print(f"\n{'#'*70}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch few-shot experiments")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Categories to test (default: quick test categories)"
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Test all 26 categories"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=50,
        help="Number of test samples per category"
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["full"],
        choices=["full", "n_shots", "strategies", "formats"],
        help="Phases to run"
    )
    
    args = parser.parse_args()
    
    categories = args.categories
    if args.all_categories:
        categories = ALL_CATEGORIES
    elif categories is None:
        categories = QUICK_TEST_CATEGORIES
    
    run_batch_experiments(
        categories=categories,
        test_size=args.test_size,
        phases=args.phases
    )


if __name__ == "__main__":
    main()
