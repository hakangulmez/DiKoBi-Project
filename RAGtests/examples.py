"""
Example usage script demonstrating the test suite.
Run this to see the full workflow.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from RAGtests.run_experiments import FewShotExperimentRunner
from RAGtests.config import DEFAULT_MODEL


def example_1_single_experiment():
    """Example 1: Run a single experiment."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Experiment")
    print("="*70 + "\n")
    
    runner = FewShotExperimentRunner(
        category="1_D_M",
        model_name=DEFAULT_MODEL,
        test_size=20,  # Small for demo
        use_8bit=True
    )
    
    try:
        result = runner.run_single_experiment(
            n_shots=3,
            selection_strategy="balanced",
            prompt_format="standard"
        )
        
        print("\n✅ Experiment completed!")
        print(f"MSE: {result['metrics']['mse']:.4f}")
        print(f"Accuracy: {result['metrics']['accuracy']:.1%}")
        print(f"Time: {result['time_seconds']:.1f}s")
        
    finally:
        runner.cleanup()


def example_2_compare_strategies():
    """Example 2: Compare selection strategies."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Compare Selection Strategies")
    print("="*70 + "\n")
    
    runner = FewShotExperimentRunner(
        category="1_D_M",
        model_name=DEFAULT_MODEL,
        test_size=20,
        use_8bit=True
    )
    
    try:
        # Test 3 different strategies
        runner.run_strategy_comparison(
            n_shots=3,
            strategies=["random", "balanced", "diverse"],
            prompt_format="standard"
        )
        
        # Results are automatically printed
        
    finally:
        runner.cleanup()


def example_3_compare_formats():
    """Example 3: Compare prompt formats."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Compare Prompt Formats")
    print("="*70 + "\n")
    
    runner = FewShotExperimentRunner(
        category="1_D_M",
        model_name=DEFAULT_MODEL,
        test_size=20,
        use_8bit=True
    )
    
    try:
        # Test 3 different formats
        runner.run_format_comparison(
            n_shots=3,
            selection_strategy="balanced",
            formats=["standard", "explicit_scale", "step_by_step"]
        )
        
        # Save results
        runner.save_results()
        
    finally:
        runner.cleanup()


def example_4_find_optimal_n_shots():
    """Example 4: Find optimal number of examples."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Find Optimal N-Shots")
    print("="*70 + "\n")
    
    runner = FewShotExperimentRunner(
        category="1_D_M",
        model_name=DEFAULT_MODEL,
        test_size=20,
        use_8bit=True
    )
    
    try:
        # Test different numbers of examples
        runner.run_n_shots_comparison(
            n_shots_list=[1, 3, 5],
            selection_strategy="balanced",
            prompt_format="standard"
        )
        
        # Save results
        runner.save_results()
        
    finally:
        runner.cleanup()


def main():
    """Run all examples."""
    print("\n" + "#"*70)
    print("FEW-SHOT PROMPTING TEST SUITE - EXAMPLES")
    print("#"*70 + "\n")
    
    print("This script demonstrates the test suite functionality.")
    print("Each example runs on a small test set (20 samples) for speed.\n")
    
    # Ask user which example to run
    print("Available examples:")
    print("  1. Single experiment")
    print("  2. Compare selection strategies")
    print("  3. Compare prompt formats")
    print("  4. Find optimal n-shots")
    print("  5. Run all examples")
    print("  0. Exit")
    
    try:
        choice = input("\nSelect example (0-5): ").strip()
        
        if choice == "1":
            example_1_single_experiment()
        elif choice == "2":
            example_2_compare_strategies()
        elif choice == "3":
            example_3_compare_formats()
        elif choice == "4":
            example_4_find_optimal_n_shots()
        elif choice == "5":
            example_1_single_experiment()
            example_2_compare_strategies()
            example_3_compare_formats()
            example_4_find_optimal_n_shots()
        elif choice == "0":
            print("Exiting...")
            return
        else:
            print("Invalid choice!")
            return
        
        print("\n" + "#"*70)
        print("EXAMPLES COMPLETED!")
        print("#"*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
