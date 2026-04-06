"""
DiKoBi Train-Test Split
========================
Splits each category dataset into 80% training and 20% test sets.

Uses stratified splitting to maintain rating distribution in both sets.
Creates separate train/ and test/ folders with matching category files.

Usage on Linux/macOS: python scripts/preprocessing/03-train-test-split.py
Usage on Windows: python scripts/preprocessing/03-train-test-split.py
"""

import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import PROCESSED_DATA_DIR
import pandas as pd
from sklearn.model_selection import train_test_split


def split_category_dataset(category_file: Path, 
                          train_folder: Path, 
                          test_folder: Path,
                          test_size: float = 0.2,
                          random_state: int = 42):
    """
    Split a single category dataset into train and test sets.
    
    Args:
        category_file: Path to category CSV file
        train_folder: Output folder for training data
        test_folder: Output folder for test data
        test_size: Fraction of data for test set (default: 0.2 = 20%)
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (train_size, test_size, category_name)
    """
    # Read the category data
    df = pd.read_csv(category_file)
    
    # Get category name (second column is the rating column with category name)
    category_col = df.columns[1]  # First is free_text_answer, second is category rating
    category_name = category_file.stem
    
    # Check if we have enough data to split
    if len(df) < 5:
        print(f"  ⚠ Skipping {category_name}: only {len(df)} samples (too few to split)")
        return None
    
    # Try stratified split by rating to maintain distribution
    try:
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df[category_col]
        )
    except ValueError:
        # If stratification fails (e.g., too few samples per class), do random split
        print(f"  ⚠ {category_name}: Using random split (not enough samples per rating for stratification)")
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state
        )
    
    # Save splits
    train_file = train_folder / category_file.name
    test_file = test_folder / category_file.name
    
    train_df.to_csv(train_file, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_file, index=False, encoding='utf-8-sig')
    
    return len(train_df), len(test_df), category_name


def main(categories_folder: str = PROCESSED_DATA_DIR / 'categories',
         output_folder: str = PROCESSED_DATA_DIR,
         test_size: float = 0.2,
         random_state: int = 42):
    """
    Split all category datasets into train and test sets.
    
    Args:
        categories_folder: Folder containing category CSV files
        output_folder: Base output folder (will create train/ and test/ subfolders)
        test_size: Fraction of data for test set (default: 0.2 = 20%)
        random_state: Random seed for reproducibility
    """
    print("=" * 70)
    print("DiKoBi Train-Test Split")
    print("=" * 70)
    print(f"Train-Test Ratio: {int((1-test_size)*100)}% / {int(test_size*100)}%")
    print(f"Random seed: {random_state}")
    print()
    
    # Create output folders
    categories_path = Path(categories_folder)
    output_path = Path(output_folder)
    train_folder = output_path / 'train'
    test_folder = output_path / 'test'
    
    train_folder.mkdir(parents=True, exist_ok=True)
    test_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all category CSV files
    category_files = sorted(categories_path.glob('*.csv'))
    category_files = [f for f in category_files if f.name != 'category_statistics.csv']
    
    if not category_files:
        print(f"✗ No category files found in: {categories_folder}")
        print(f"\nPlease run the category splitting script first:")
        print(f"  python scripts/preprocessing/02-split-categories.py")
        return []
    
    print(f"Found {len(category_files)} category files")
    print("-" * 70)
    
    # Split each category
    results = []
    total_train = 0
    total_test = 0
    skipped = 0
    
    for category_file in category_files:
        result = split_category_dataset(
            category_file, 
            train_folder, 
            test_folder,
            test_size=test_size,
            random_state=random_state
        )
        
        if result:
            train_size, test_count, category_name = result
            results.append({
                'category': category_name,
                'train_samples': train_size,
                'test_samples': test_count,
                'total_samples': train_size + test_count,
                'test_percentage': (test_count / (train_size + test_count)) * 100
            })
            total_train += train_size
            total_test += test_count
            print(f"  ✓ {category_name:15s}: {train_size:4d} train / {test_count:4d} test")
        else:
            skipped += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Categories processed: {len(results)}")
    print(f"Categories skipped: {skipped}")
    print(f"Total train samples: {total_train}")
    print(f"Total test samples: {total_test}")
    if (total_train + total_test) == 0:
        print("Overall split: N/A (no samples processed)")
    else:
        print(f"Overall split: {(total_train/(total_train+total_test))*100:.1f}% / {(total_test/(total_train+total_test))*100:.1f}%")
    
    print(f"\n✓ Train data saved to: {train_folder}")
    print(f"✓ Test data saved to: {test_folder}")
    
    # Preserve category_statistics.csv before cleaning up
    stats_file = categories_path / 'category_statistics.csv'
    if stats_file.exists():
        preserved_stats = output_path / 'category_statistics.csv'
        shutil.copy2(stats_file, preserved_stats)
        print(f"✓ Preserved statistics to: {preserved_stats}")
    else:
        print(f"ℹ️  No statistics file found at: {stats_file}")
    
    # Clean up temporary categories folder
    if categories_path.exists() and categories_path.name == 'categories':
        try:
            shutil.rmtree(categories_path)
            print(f"✓ Cleaned up temporary folder: {categories_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not clean up temporary folder {categories_path}: {e}")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        
        if results:
            print("\n✓ Train-test split completed successfully!")
        else:
            print("\n✗ Failed to create train-test split")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
