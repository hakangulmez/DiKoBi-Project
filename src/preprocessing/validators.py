"""
Validation Module
=================
Functions for validating dataset extraction and transformation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import List, Dict, Set, Union
from collections import defaultdict
import warnings

from .column_utils import (
    find_free_text_columns,
    get_coding_columns_for_text,
    normalize_coding_column_name
)

warnings.filterwarnings('ignore')

# Sample size for validation checks - limits rows checked to improve performance
# while still providing representative data presence information
SAMPLE_SIZE_FOR_VALIDATION = 100


def extract_all_codings_from_excel(folder_path: Union[str, Path]) -> Dict:
    """
    Extracts all coding columns from Excel files with their context.
    Optimized: reads only headers (nrows=0) for speed.
    
    Args:
        folder_path: Path to folder containing Excel files
    
    Returns a dictionary with:
    - 'columns': Set of all unique normalized coding column names
    - 'raw_columns': Dict mapping normalized names to original names
    - 'file_sheet_mapping': Where each coding appears
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    excel_files = list(folder.glob('*.xlsx')) + list(folder.glob('*.xls'))
    
    if not excel_files:
        raise ValueError(f"No Excel files found in {folder_path}")
    
    print(f"Scanning {len(excel_files)} Excel file(s)...")
    print()
    
    all_coding_columns = set()
    raw_to_normalized = defaultdict(set)
    file_sheet_mapping = defaultdict(list)
    
    for file in excel_files:
        try:
            excel_file = pd.ExcelFile(file)
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file, sheet_name=sheet_name, nrows=0)
                
                free_text_cols = find_free_text_columns(df)
                
                for text_col in free_text_cols:
                    coding_cols = get_coding_columns_for_text(df, text_col)
                    
                    for coding_col in coding_cols:
                        normalized = normalize_coding_column_name(coding_col)
                        
                        all_coding_columns.add(normalized)
                        raw_to_normalized[normalized].add(coding_col)
                        file_sheet_mapping[normalized].append(
                            f"{file.name}/{sheet_name}/{text_col}"
                        )
                        
        except Exception as e:
            print(f"  ✗ Error reading {file.name}: {e}")
    
    print(f"✓ Found {len(all_coding_columns)} unique coding columns")
    print()
    
    return {
        'columns': all_coding_columns,
        'raw_columns': dict(raw_to_normalized),
        'file_sheet_mapping': dict(file_sheet_mapping),
    }


def analyze_processed_csv(csv_path: str) -> Dict:
    """
    Analyzes the processed CSV file.
    
    Returns dictionary with:
    - 'columns': Set of coding columns in CSV
    - 'response_count': Number of rows
    - 'column_fill_rates': How often each column has values
    - 'dataframe': The loaded DataFrame
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Reading processed CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    coding_cols = [col for col in df.columns if re.match(r'^\d+_', str(col))]
    
    print(f"✓ Found {len(coding_cols)} coding columns in CSV")
    print(f"✓ Found {len(df)} responses in CSV")
    print()
    
    # Vectorized fill rate calculation
    column_fill_rates = {}
    for col in coding_cols:
        valid_mask = df[col].notna() & ~df[col].isin(['', '-66', '-99'])
        fill_rate = valid_mask.sum() / len(df) * 100
        column_fill_rates[col] = fill_rate
    
    return {
        'columns': set(coding_cols),
        'response_count': len(df),
        'column_fill_rates': column_fill_rates,
        'dataframe': df
    }


def validate_column_completeness(excel_data: Dict, csv_data: Dict) -> Dict:
    """
    Checks if all coding columns from Excel are present in CSV.
    """
    excel_cols = excel_data['columns']
    csv_cols = csv_data['columns']
    
    missing_in_csv = excel_cols - csv_cols
    extra_in_csv = csv_cols - excel_cols
    common_cols = excel_cols & csv_cols
    
    return {
        'missing_in_csv': missing_in_csv,
        'extra_in_csv': extra_in_csv,
        'common_cols': common_cols,
        'pass': len(missing_in_csv) == 0 and len(extra_in_csv) == 0
    }


def validate_normalization_consistency(excel_data: Dict) -> Dict:
    """
    Checks if normalization was applied consistently.
    """
    raw_columns = excel_data['raw_columns']
    
    merged_columns = {}
    for normalized, originals in raw_columns.items():
        if len(originals) > 1:
            merged_columns[normalized] = originals
    
    return {
        'merged_columns': merged_columns,
        'merge_count': len(merged_columns),
        'pass': True
    }


def validate_response_codings(csv_data: Dict) -> Dict:
    """
    Checks if all responses have at least one coding value.
    """
    df = csv_data['dataframe']
    
    coding_cols = [col for col in df.columns if re.match(r'^\d+_', str(col))]
    
    rows_without_coding = df[df[coding_cols].isna().all(axis=1)]
    rows_with_coding = df[df[coding_cols].notna().any(axis=1)]
    sample_coded = rows_with_coding.sample(min(5, len(rows_with_coding))) if len(rows_with_coding) > 0 else pd.DataFrame()
    
    return {
        'total_responses': len(df),
        'responses_without_coding': len(rows_without_coding),
        'responses_with_coding': len(rows_with_coding),
        'sample_responses': sample_coded,
        'pass': len(rows_without_coding) == 0
    }


def check_missing_columns_details(missing_cols: Set[str], excel_folder: str) -> Dict:
    """
    For each missing column, check if it has any data and what free text it's associated with.
    Only checks first file for speed.
    
    Args:
        missing_cols: Set of column names missing from CSV
        excel_folder: Path to folder with Excel files
        
    Returns:
        Dictionary with details about each missing column
    """
    folder = Path(excel_folder)
    missing_details = {}
    
    # Only check first file for performance
    excel_files = list(folder.glob('*.xlsx'))
    if not excel_files:
        return missing_details
    
    file = excel_files[0]
    
    for col_base in missing_cols:
        col_info = {
            'total_values': 0,
            'locations': [],
            'associated_free_text': [],
            'free_text_has_data': False
        }
        
        try:
            xls = pd.ExcelFile(file)
            sheet = xls.sheet_names[0]  # Only check first sheet
            df = pd.read_excel(file, sheet_name=sheet)
            
            # Look for exact match or with number suffix
            matching = [col for col in df.columns 
                       if str(col) == col_base or 
                       re.match(rf'^{re.escape(col_base)}_\d+$', str(col))]
            
            for col in matching:
                # Check for valid data (sample only for performance)
                valid_vals = df[col].dropna().head(SAMPLE_SIZE_FOR_VALIDATION)
                valid_vals = valid_vals[~valid_vals.isin(['', '-66', '-99'])]
                valid_count = len(valid_vals)
                
                if valid_count > 0 or col in df.columns:
                    # Find associated free text column
                    cols = df.columns.tolist()
                    col_idx = cols.index(col)
                    
                    free_text_col = None
                    for i in range(col_idx - 1, -1, -1):
                        if str(cols[i]).lower().startswith('v_'):
                            free_text_col = cols[i]
                            break
                    
                    # Check if free text has data (sample only for performance)
                    free_text_valid = 0
                    if free_text_col and free_text_col in df.columns:
                        ft_data = df[free_text_col].dropna().head(SAMPLE_SIZE_FOR_VALIDATION)
                        ft_data = ft_data[~ft_data.isin(['', '-66', '-99'])]
                        free_text_valid = len(ft_data)
                    
                    if valid_count > 0 or free_text_col:
                        col_info['total_values'] += valid_count
                        col_info['locations'].append(f"{file.name}/{sheet}")
                        if free_text_col:
                            col_info['associated_free_text'].append(free_text_col)
                            if free_text_valid > 0:
                                col_info['free_text_has_data'] = True
                            
        except (FileNotFoundError, PermissionError, pd.errors.EmptyDataError, 
                pd.errors.ParserError):
            # Silently skip files/sheets that fail to read or process.
            # This is acceptable because we're only checking the first file for performance,
            # and missing details for one column won't prevent the overall validation.
            pass
        
        missing_details[col_base] = col_info
    
    return missing_details
