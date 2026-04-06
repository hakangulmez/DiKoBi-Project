"""
Column Identification and Normalization Utilities
==================================================
Helper functions for identifying and normalizing DiKobi dataset columns.
"""

import pandas as pd
import re
from typing import List


def find_free_text_columns(df: pd.DataFrame) -> List[str]:
    """
    Identifies columns containing free text answers.
    
    Free text columns start with 'v_' or 'v ' (case insensitive).
    
    Args:
        df: DataFrame with column headers
        
    Returns:
        List of column names that are free text responses
    """
    free_text_cols = [col for col in df.columns 
                     if str(col).lower().startswith('v_') or str(col).lower().startswith('v ')]
    return free_text_cols


def get_coding_columns_for_text(df: pd.DataFrame, text_col: str) -> List[str]:
    """
    Identifies all coding columns associated with a specific free text column.
    
    Coding columns follow their corresponding free text column until the 
    next v_ column is encountered.
    
    Args:
        df: DataFrame with column headers
        text_col: Name of the free text column
        
    Returns:
        List of coding column names associated with this text column
    """
    cols = df.columns.tolist()
    
    try:
        text_idx = cols.index(text_col)
    except ValueError:
        return []
    
    coding_cols = []
    for i in range(text_idx + 1, len(cols)):
        col = cols[i]
        # Stop at next free text column
        if str(col).lower().startswith('v_') or str(col).lower().startswith('v '):
            break
        # Only include columns that start with a digit (standard coding format)
        col_str = str(col)
        if not col_str.startswith('Unnamed') and re.match(r'^\d+_', col_str):
            coding_cols.append(col)
    
    return coding_cols


def normalize_coding_column_name(column_name: str) -> str:
    """
    Removes trailing item numbers from coding column names for consistency.
    
    Item numbers (numeric suffixes like _1, _2, _9) are removed to create
    a unified category name, while alphabetic subcategories are preserved.
    
    Examples:
        '5_D_Ma_2' -> '5_D_Ma'
        '1_D_kA_1' -> '1_D_kA'
        '4_D_PK_7' -> '4_D_PK'
        '1_Dm_kA' -> '1_Dm_kA' (no change, kA is subcategory)
        '1_Dm_M' -> '1_Dm_M' (no change, M is subcategory)
    
    Args:
        column_name: Original column name
        
    Returns:
        Normalized column name without numeric item suffix
    """
    col_str = str(column_name)
    
    # Only normalize columns starting with a digit (coding columns)
    if not re.match(r'^\d+_', col_str):
        return column_name
    
    # Remove trailing numeric item number (e.g., _1, _2, _9)
    # But preserve alphabetic subcategories (e.g., kA, Ma, PCK)
    normalized = re.sub(r'_(\d+(?:\.\d+)?)$', '', col_str)
    return normalized


def identify_metadata_columns() -> List[str]:
    """
    Returns standard metadata column names.
    
    Returns:
        List of metadata column names
    """
    return ['response_id', 'participant_id', 'source_file', 'source_sheet', 
            'free_text_column', 'free_text_answer']


def identify_category_columns(df: pd.DataFrame) -> List[str]:
    """
    Identifies all coding category columns (start with digit + underscore).
    
    Args:
        df: DataFrame with column headers
        
    Returns:
        Sorted list of category column names
    """
    return sorted([col for col in df.columns 
                   if re.match(r'^\d+_', str(col))])
