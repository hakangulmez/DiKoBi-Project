"""
DiKoBi Complete Preprocessing Pipeline
========================================
Runs all preprocessing steps in sequence:
1. Prepare dataset (Excel → long format CSV)
2. Split into categories
3. Create train/test splits (80/20)

This is the main script to run when starting with raw Excel files.

Usage on Linux/macOS: python scripts/preprocessing/00-preprocess-all.py
Usage on Windows: python scripts/preprocessing/00-preprocess-all.py
"""

import importlib.util
import sys
import importlib.util
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import PROCESSED_DATA_DIR


def run_preprocessing_step(script_name: str, step_num: int, description: str, 
                          check_nonempty: bool = False) -> bool:
    """Execute a preprocessing step from a script file.
    
    Args:
        script_name: Name of the script file to execute
        step_num: Step number for display purposes
        description: Description of the step for display
        check_nonempty: If True, also check that result is not empty/None
        
    Returns:
        bool: True if step succeeded, False otherwise
    """
    print(f"\nSTEP {step_num}: {description}...")
    print("=" * 70)
    try:
        spec = importlib.util.spec_from_file_location(
            script_name.replace('-', '_'),
            Path(__file__).parent / script_name
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.main()
        
        # Handle different success criteria
        if check_nonempty:
            # Check that result exists and is not empty (for collections)
            if result is None:
                success = False
            elif hasattr(result, '__len__'):
                success = len(result) > 0
            else:
                success = True  # Non-collection non-None values are considered successful
        else:
            success = bool(result)
            
        if not success:
            print(f"\n✗ Step {step_num} failed: script returned unsuccessful status")
            return False
            
        print(f"\n✓ Step {step_num} completed\n")
        return True
    except Exception as e:
        print(f"\n✗ Step {step_num} failed: {e}")
        return False


def main():
    """
    Execute complete preprocessing pipeline.
    """
    print("=" * 70)
    print("DiKoBi Complete Preprocessing Pipeline")
    print("=" * 70)
    print()
    
    # Step 1: Prepare dataset
    if not run_preprocessing_step("01-prepare-dataset.py", 1, "Preparing dataset", check_nonempty=True):
        return False
    
    # Step 2: Split categories
    if not run_preprocessing_step("02-split-categories.py", 2, "Splitting into categories"):
        return False
    
    # Step 3: Train-test split
    if not run_preprocessing_step("03-train-test-split.py", 3, "Creating train/test splits"):
        return False
    
    # Final summary
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"✓ Long format data: {PROCESSED_DATA_DIR}/dikobi_long_format.csv")
    print(f"✓ Training data: {PROCESSED_DATA_DIR}/train/")
    print(f"✓ Test data: {PROCESSED_DATA_DIR}/test/")
    print()
    print("Ready for model training and evaluation!")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
