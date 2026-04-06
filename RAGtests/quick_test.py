"""
Quick test script for rapid experimentation.
Use this for quick tests before running full experiment suite.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RAGtests.run_experiments import FewShotExperimentRunner
from RAGtests.config import QUICK_TEST_CATEGORIES, DEFAULT_MODEL


def quick_test(
    category: str = "1_D_M",
    n_shots: int = 3,
    strategy: str = "balanced",
    format_type: str = "standard",
    test_size: int = 20  # Very small for quick testing
):
    """
    Run a quick single experiment for testing.
    
    Args:
        category: Category to test
        n_shots: Number of examples
        strategy: Selection strategy
        format_type: Prompt format
        test_size: Number of test samples
    """
    print(f"\n{'#'*70}")
    print(f"QUICK TEST")
    print(f"{'#'*70}")
    print(f"Category: {category}")
    print(f"N-Shots: {n_shots}")
    print(f"Strategy: {strategy}")
    print(f"Format: {format_type}")
    print(f"Test Size: {test_size}")
    print(f"{'#'*70}\n")
    
    # Create runner
    runner = FewShotExperimentRunner(
        category=category,
        model_name=DEFAULT_MODEL,
        use_8bit=True,
        test_size=test_size
    )
    
    try:
        # Run single experiment
        result = runner.run_single_experiment(
            n_shots=n_shots,
            selection_strategy=strategy,
            prompt_format=format_type
        )
        
        print(f"\n✅ Quick test completed!")
        print(f"MSE: {result['metrics']['mse']:.4f}")
        print(f"Accuracy: {result['metrics']['accuracy']:.1%}")
        
    finally:
        runner.cleanup()


def test_all_strategies(category: str = "1_D_M", test_size: int = 20):
    """Quick test of all selection strategies."""
    runner = FewShotExperimentRunner(
        category=category,
        model_name=DEFAULT_MODEL,
        use_8bit=True,
        test_size=test_size
    )
    
    try:
        runner.run_strategy_comparison(
            n_shots=3,
            strategies=["random", "balanced", "stratified", "diverse"],
            prompt_format="standard"
        )
        runner.save_results()
    finally:
        runner.cleanup()


def test_all_formats(category: str = "1_D_M", test_size: int = 20):
    """Quick test of all prompt formats."""
    runner = FewShotExperimentRunner(
        category=category,
        model_name=DEFAULT_MODEL,
        use_8bit=True,
        test_size=test_size
    )
    
    try:
        runner.run_format_comparison(
            n_shots=3,
            selection_strategy="balanced"
        )
        runner.save_results()
    finally:
        runner.cleanup()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick few-shot testing")
    parser.add_argument("--category", default="1_D_M", help="Category to test")
    parser.add_argument("--n-shots", type=int, default=3, help="Number of examples")
    parser.add_argument("--strategy", default="balanced", help="Selection strategy")
    parser.add_argument("--format", default="standard", help="Prompt format")
    parser.add_argument("--test-size", type=int, default=20, help="Number of test samples")
    parser.add_argument("--test-strategies", action="store_true", help="Test all strategies")
    parser.add_argument("--test-formats", action="store_true", help="Test all formats")
    
    args = parser.parse_args()
    
    if args.test_strategies:
        test_all_strategies(args.category, args.test_size)
    elif args.test_formats:
        test_all_formats(args.category, args.test_size)
    else:
        quick_test(
            category=args.category,
            n_shots=args.n_shots,
            strategy=args.strategy,
            format_type=args.format,
            test_size=args.test_size
        )
