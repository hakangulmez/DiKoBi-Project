"""
Category Dataset Extraction Module
===================================
Extracts category-specific datasets for ML/prompting applications.

Each category dataset contains only:
- free_text_answer: The free text response
- rating: The integer rating for that specific category

Responses without a valid rating for a category are excluded from that category's dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Union
import re


def identify_category_columns(df: pd.DataFrame) -> List[str]:
    """
    Identifies all coding category columns in the dataset.
    
    Category columns start with a digit followed by underscore (e.g., 1_D_M, 2_D_PCK).
    
    Args:
        df: DataFrame with normalized column names
        
    Returns:
        List of category column names sorted alphabetically
    """
    category_cols = []
    
    for col in df.columns:
        col_str = str(col)
        # Match coding columns: start with digit + underscore
        if re.match(r'^\d+_', col_str):
            category_cols.append(col)
    
    return sorted(category_cols)


def extract_category_dataset(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Extracts a dataset for a specific category.
    
    Creates a DataFrame with only:
    - free_text_answer: The response text
    - {category}: The integer rating for this category
    
    Only includes rows where the category has a valid (non-null) rating.
    
    Args:
        df: Long format DataFrame with all categories
        category: Name of the category column to extract
        
    Returns:
        DataFrame with free_text_answer and category rating columns
    """
    if category not in df.columns:
        raise ValueError(f"Category '{category}' not found in dataset")
    
    if 'free_text_answer' not in df.columns:
        raise ValueError("Column 'free_text_answer' not found in dataset")
    
    # Filter rows with valid ratings for this category
    valid_mask = df[category].notna()
    filtered_df = df[valid_mask].copy()
    
    # Create category-specific dataset with category name as column
    category_df = pd.DataFrame({
        'free_text_answer': filtered_df['free_text_answer'],
        category: filtered_df[category].astype(int)
    })
    
    return category_df


def create_all_category_datasets(input_file: Union[str, Path], output_folder: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Creates separate dataset files for each category.
    
    Args:
        input_file: Path to the long format CSV file
        output_folder: Folder where category datasets will be saved
        
    Returns:
        Dictionary mapping category names to their DataFrames
    """
    # Read the long format dataset
    print(f"Reading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df)} responses")
    
    # Create output folder
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Identify all category columns
    categories = identify_category_columns(df)
    print(f"\n✓ Found {len(categories)} categories")
    
    # Extract and save each category dataset
    category_datasets = {}
    
    print("\nCreating category datasets:")
    print("-" * 70)
    
    for category in categories:
        try:
            # Extract category-specific data
            category_df = extract_category_dataset(df, category)
            
            # Skip empty categories
            if len(category_df) == 0:
                print(f"  ⚠ {category}: No valid ratings - skipped")
                continue
            
            # Save to CSV
            output_file = output_path / f"{category}.csv"
            category_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            category_datasets[category] = category_df
            
            # Summary statistics
            rating_counts = category_df[category].value_counts().sort_index()
            rating_summary = ', '.join([f"{int(val)}:{count}" for val, count in rating_counts.items()])
            
            print(f"  ✓ {category}: {len(category_df)} responses [{rating_summary}]")
            
        except Exception as e:
            print(f"  ✗ {category}: Error - {e}")
    
    print(f"\n✓ Created {len(category_datasets)} category datasets in: {output_folder}")
    
    return category_datasets


def get_category_statistics(category_datasets: Dict[str, pd.DataFrame], source_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generates statistics for all category datasets.
    
    Args:
        category_datasets: Dictionary mapping category names to DataFrames
        source_df: Optional source DataFrame to get coding information
        
    Returns:
        DataFrame with statistics for each category including coding distributions
    """
    stats = []
    
    for category, df in category_datasets.items():
        # Get the rating column (second column, which is the category name)
        rating_col = df.columns[1]
        rating_counts = df[rating_col].value_counts().sort_index()
        
        stat = {
            'category': category,
            'total_responses': len(df),
            'max_rating': df[rating_col].max(),
            'rating_distribution': dict(rating_counts)
        }
        
        # Add coding distribution if source dataframe provided
        if source_df is not None and 'coding' in source_df.columns:
            # Filter source_df for rows that have valid ratings for this category
            valid_mask = source_df[category].notna()
            coding_counts = source_df[valid_mask]['coding'].value_counts().sort_index()
            stat['coding_distribution'] = dict(coding_counts)
        else:
            stat['coding_distribution'] = {}
        
        stats.append(stat)
    
    stats_df = pd.DataFrame(stats)
    return stats_df
