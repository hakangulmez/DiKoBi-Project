"""
DiKoBi Category Dataset Creator
=================================
Splits the long format dataset into separate files for each category.

Each category dataset contains:
- free_text_answer: The free text response
- rating: The integer rating for that category

Only responses with valid ratings for a category are included in that category's dataset.

Usage on Linux/macOS: python scripts/preprocessing/02-split-categories.py
Usage on Windows: python scripts/preprocessing/02-split-categories.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import PROCESSED_DATA_DIR
from src.preprocessing.category_splitter import (
    create_all_category_datasets,
    get_category_statistics
)
import pandas as pd


def main(input_file: str = PROCESSED_DATA_DIR / 'dikobi_long_format.csv',
         output_folder: str = PROCESSED_DATA_DIR / 'categories'):
    """
    Creates separate CSV files for each category in the dataset.
    
    Each category file contains only the free text answers and their ratings
    for that specific category (with the category name as the column header),
    making them ready for ML/prompting applications.
    
    Args:
        input_file: Path to long format CSV (default: data/processed/dikobi_long_format.csv)
        output_folder: Destination folder (default: data/processed/categories/)
    """
    print("=" * 70)
    print("DiKoBi Category Dataset Creator")
    print("=" * 70)
    print()
    
    # Check if input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"✗ Error: Input file not found: {input_file}")
        print(f"\nPlease run the data preparation script first:")
        print(f"  python scripts/prepare_dataset.py")
        return None
    
    # Create category datasets
    try:
        category_datasets = create_all_category_datasets(input_file, output_folder)
        
        if not category_datasets:
            print("\n✗ No category datasets were created")
            return None
        
        # Read the source data for coding statistics
        source_df = pd.read_csv(input_file)
        
        # Generate and display statistics
        print("\n" + "=" * 70)
        print("CATEGORY STATISTICS")
        print("=" * 70)
        
        stats_df = get_category_statistics(category_datasets, source_df)
        
        # Display summary statistics
        print(f"\nTotal categories created: {len(category_datasets)}")
        print(f"Total responses across all categories: {stats_df['total_responses'].sum()}")
        print(f"\nTop 10 categories by number of responses:")
        print("-" * 70)
        
        top_categories = stats_df.nlargest(10, 'total_responses')
        for _, row in top_categories.iterrows():
            ratings_str = ', '.join([f"{k}:{v}" for k, v in sorted(row['rating_distribution'].items())])
            codings_str = ', '.join([f"{k}:{v}" for k, v in sorted(row['coding_distribution'].items())])
            print(f"  {row['category']:15s} : {row['total_responses']:4d} responses")
            print(f"    Ratings: [{ratings_str}]")
            print(f"    Codings: [{codings_str}]")
        
        # Save detailed statistics
        stats_output = Path(output_folder) / 'category_statistics.csv'
        # Create separate columns for each rating and coding count
        stats_export = stats_df.drop(columns=['rating_distribution', 'coding_distribution']).copy()
        
        # Add rating columns
        all_ratings = sorted(set(r for dist in stats_df['rating_distribution'] for r in dist.keys()))
        for rating in all_ratings:
            stats_export[f'rating_{rating}'] = stats_df['rating_distribution'].apply(lambda x: x.get(rating, 0))
        
        # Add coding columns
        all_codings = sorted(set(c for dist in stats_df['coding_distribution'] for c in dist.keys()))
        for coding in all_codings:
            stats_export[f'coding_{coding}'] = stats_df['coding_distribution'].apply(lambda x: x.get(coding, 0))
        
        stats_export.to_csv(stats_output, index=False, encoding='utf-8-sig')
        print(f"\n✓ Detailed statistics saved to: {stats_output}")
        
        return category_datasets
        
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    try:
        category_datasets = main()
        
        if category_datasets:
            print("\n✓ Category datasets created successfully!")
        else:
            print("\n✗ Failed to create category datasets")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
