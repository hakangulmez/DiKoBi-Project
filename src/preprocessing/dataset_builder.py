"""
Dataset Creation Module
========================
Converts Excel data to long format with associated coding ratings.

Architecture:
    1. VALUE CLEANING: Validates and cleans individual coding values
    2. DATA LOADING: Reads Excel files and extracts responses
       - read_excel_files_to_long_format(): Orchestrates file processing (Entry point)
       - process_sheet_to_long_format(): Processes one sheet
       - Helper functions: Extract participant info, validate responses
    3. COLUMN OPERATIONS: Normalizes, cleans, and sorts columns

Based on the DiKoBi coding manual for assessing diagnostic competencies
of biology teachers using video-based simulations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import List, Union
import warnings

from .standards import MAX_VALUES
from .column_utils import (
    find_free_text_columns,
    get_coding_columns_for_text,
    normalize_coding_column_name,
    identify_metadata_columns
)

warnings.filterwarnings('ignore')


# ============================================================================
# Constants
# ============================================================================

# Special values indicating missing data
MISSING_DATA_CODES = {'', '-66', '-99', 'nan'}

# Files to skip due to problematic structure
SKIP_FILES = {'C1_S3_Pilotierung.xlsx', 'C1_S3_Pre_Int_Post.xlsx'}


# ============================================================================
# Helper Functions: VALUE CLEANING AND VALIDATION
# ============================================================================


def clean_coding_value(value, column_name: str = None, violation_tracker: dict = None) -> int:
    """
    Converts coding values to integers, handling missing values, error codes,
    and values exceeding maximum allowed values.
    
    Args:
        value: Raw value from coding column
        column_name: Optional column name for validation (unnormalized ok, e.g. '1_D_kA_2')
        violation_tracker: Optional dict to count violations
        
    Returns:
        Integer coding value or NaN for invalid/missing values
    """
    if pd.isna(value):
        return np.nan
    
    value_str = str(value).strip()
    
    # Handle missing data and error codes
    if value_str in MISSING_DATA_CODES:
        return np.nan
    
    try:
        # Convert to float first, then check if it's a whole number
        float_val = float(value_str)
        if float_val != int(float_val):
            return np.nan
        
        int_val = int(float_val)
        
        # Validate against maximum if column name provided
        if column_name:
            normalized = normalize_coding_column_name(column_name)
            max_val = MAX_VALUES.get(normalized, float('inf'))
            
            if max_val != float('inf') and int_val > max_val:
                if violation_tracker is not None:
                    violation_tracker['count'] = violation_tracker.get('count', 0) + 1
                return np.nan
        
        return int_val
        
    except (ValueError, TypeError):
        return np.nan


# ============================================================================
# Helper Functions: DATA LOADING AND TRANSFORMATION
# ============================================================================

def _extract_participant_info(row: pd.Series, id_columns: List[str], source_file: str, source_sheet: str) -> dict:
    """Extracts participant metadata from a row."""
    participant_info = {col: row[col] for col in id_columns}
    
    # Normalize 'Code' to 'participant_id' for consistency
    if 'Code' in participant_info:
        participant_info['participant_id'] = participant_info.pop('Code')
    
    participant_info['source_file'] = source_file
    participant_info['source_sheet'] = source_sheet
    
    return participant_info


def _normalize_question_type(text_col: str) -> str:
    """
    Extracts and normalizes question type from column name.
    
    Examples:
        'v_940 (Beschreibe)' -> 'describe'
        'v_977 (Begründe 1)' -> 'explain'
    """
    question_type_match = re.search(r'\(([^)]+)\)', text_col)
    if not question_type_match:
        return 'unknown'
    
    question_type_raw = question_type_match.group(1)
    
    # Remove trailing numbers and spaces, convert to lowercase
    qt_clean = re.sub(r'\s+\d+$', '', question_type_raw).strip().lower()
    
    # Map German variants to English
    if qt_clean.startswith('beschreib'):  # Beschreibe, Beschreiben, Beschreibung
        return 'describe'
    elif qt_clean.startswith('begründ'):  # Begründe, Begründen, Begründung
        return 'explain'
    else:
        return question_type_raw  # Keep original if no match


def _has_valid_coding(row: pd.Series, coding_cols: List[str]) -> bool:
    """Checks if row has at least one valid (non-null) coding value."""
    for coding_col in coding_cols:
        cleaned_value = clean_coding_value(row[coding_col])
        if pd.notna(cleaned_value):
            return True
    return False


def _process_free_text_response(row: pd.Series, text_col: str, df: pd.DataFrame, 
                                 participant_info: dict, violation_tracker: dict = None) -> dict:
    """
    Processes a single free text response and its associated codings.
    
    Returns:
        Dictionary containing response data, or None if response should be skipped
    """
    # Get the free text answer
    free_text = row[text_col]
    
    # Skip missing responses and error codes
    if pd.isna(free_text):
        return None
    
    free_text_str = str(free_text).strip()
    if free_text_str in MISSING_DATA_CODES:
        return None
    
    # Get coding columns for this free text
    coding_cols = get_coding_columns_for_text(df, text_col)
    if not coding_cols:
        return None
    
    # Skip responses without any valid codings
    if not _has_valid_coding(row, coding_cols):
        return None
    
    # Build response record
    record = participant_info.copy()
    record['free_text_column'] = text_col
    record['question_type'] = _normalize_question_type(text_col)
    record['free_text_answer'] = free_text
    
    # Add coding values
    for coding_col in coding_cols:
        normalized_name = normalize_coding_column_name(coding_col)
        cleaned_value = clean_coding_value(row[coding_col], coding_col, violation_tracker)
        if pd.notna(cleaned_value):
            record[normalized_name] = cleaned_value
    
    return record


# ============================================================================
# DATA LOADING AND TRANSFORMATION
# ============================================================================
#
# Architecture:
#   read_excel_files_to_long_format()      [Entry point - processes all files]
#       └─> process_sheet_to_long_format() [Processes one sheet]
#               ├─> _extract_participant_info()
#               └─> _process_free_text_response()
#                       ├─> _normalize_question_type()
#                       ├─> _has_valid_coding()
#                       └─> clean_coding_value()
#
# ============================================================================

# --- Main Functions ---

def read_excel_files_to_long_format(folder_path: Union[str, Path]) -> pd.DataFrame:
    """
    Main entry point: Reads all Excel files and converts to long format.
    
    This function orchestrates the entire extraction process by:
    - Finding all Excel files in folder
    - Processing each file's first sheet
    - Combining all responses into single DataFrame
    - Adding sequential response IDs
    
    Args:
        folder_path: Path to folder containing Excel files
        
    Returns:
        Long format DataFrame with one row per free text response
        
    Note:
        Calls process_sheet_to_long_format() for each sheet.
        Problematic files in SKIP_FILES are automatically excluded.
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Find all Excel files
    excel_files = list(folder.glob('*.xlsx')) + list(folder.glob('*.xls'))
    
    if not excel_files:
        raise ValueError(f"No Excel files found in {folder_path}")
    
    print(f"Found {len(excel_files)} Excel file(s):")
    for f in excel_files:
        print(f"  - {f.name}")
    
    all_responses = []
    violation_tracker = {'count': 0}
    
    for file in excel_files:
        # Skip problematic files
        if file.name in SKIP_FILES:
            print(f"\n  Skipping {file.name} (problematic file)")
            continue
            
        try:
            # Read only the first sheet from the Excel file (other sheets may be empty/problematic)
            excel_file = pd.ExcelFile(file)
            sheet_names = excel_file.sheet_names

            if not sheet_names:
                print(f"\n  {file.name} has no sheets, skipping.")
                continue

            first_sheet = sheet_names[0]
            print(f"\n  {file.name} — reading in first sheet: '{first_sheet}' (of {len(sheet_names)} total)")

            try:
                df = pd.read_excel(file, sheet_name=first_sheet)
            except Exception as e:
                print(f"    ✗ Error reading sheet '{first_sheet}' in {file.name}: {e}")
                import traceback
                traceback.print_exc()
                continue

            print(f"    Reading sheet '{first_sheet}': {len(df)} participants...")

            # Convert this sheet directly to long format
            sheet_responses = process_sheet_to_long_format(
                df,
                source_file=file.name,
                source_sheet=first_sheet,
                violation_tracker=violation_tracker
            )

            all_responses.extend(sheet_responses)
            print(f"      ✓ Extracted {len(sheet_responses)} responses")
                
        except Exception as e:
            print(f"  ✗ Error reading {file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_responses:
        raise ValueError("No responses could be extracted from Excel files")
    
    # Create DataFrame - Each response dictionary contains only its own coding columns
    long_df = pd.DataFrame(all_responses)
    
    # Sequential response IDs for tracking
    long_df.insert(0, 'response_id', range(1, len(long_df) + 1))
    
    print(f"\n✓ Extracted {len(long_df)} total responses")
    print(f"  Total columns: {len(long_df.columns)}")
    
    if violation_tracker['count'] > 0:
        print(f"  ✓ Removed {violation_tracker['count']} coding value(s) exceeding maximum allowed values")
    
    return long_df


def process_sheet_to_long_format(df: pd.DataFrame, source_file: str, source_sheet: str, 
                                 violation_tracker: dict = None) -> List[dict]:
    """
    Processes a single Excel sheet, extracting all free text responses.
    
    Called by read_excel_files_to_long_format() for each sheet. This function:
    - Identifies participant IDs and free text columns
    - Iterates through each participant's responses
    - Extracts valid responses with their codings
    - Returns list of response dictionaries
    
    Each response dictionary contains only its associated coding columns,
    preventing column explosion (where pandas creates columns for all
    possible codings across all responses).
    
    Args:
        df: DataFrame from one Excel sheet
        source_file: Name of source file for tracking
        source_sheet: Name of source sheet for tracking
        violation_tracker: Optional dict to track coding value violations
        
    Returns:
        List of dictionaries, one per free text response with valid codings
        
    Note:
        Uses helper functions: _extract_participant_info(), 
        _process_free_text_response()
    """
    # Special case: Rename 4_E_IV → 4_E_T_IV for consistency
    # In C1_S1_Post.xlsx, item _2 uses 4_E_IV while others use 4_E_T_IV
    rename_map = {col: str(col).replace('4_E_IV', '4_E_T_IV', 1) 
                  for col in df.columns if str(col).startswith('4_E_IV')}
    if rename_map:
        df = df.rename(columns=rename_map)
    
    # Identify participant ID columns
    id_columns = [col for col in ['Code', 'participant_id', 'id'] if col in df.columns]
    if not id_columns:
        # Create participant ID if none exists
        df['participant_id'] = range(1, len(df) + 1)
        id_columns = ['participant_id']
    
    # Find free text columns
    free_text_cols = find_free_text_columns(df)
    if not free_text_cols:
        return []
    
    # Process each participant's responses
    responses = []
    for idx, row in df.iterrows():
        participant_info = _extract_participant_info(row, id_columns, source_file, source_sheet)
        
        # Process each free text response for this participant
        for text_col in free_text_cols:
            record = _process_free_text_response(row, text_col, df, participant_info, violation_tracker)
            if record is not None:
                responses.append(record)
    
    return responses


def normalize_dataset_columns(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes coding column names and merges duplicates.
    
    Special case: 4_E_IV is merged with 4_E_T_IV because the column 4_E_IV 
    appears only in file C1_S1_Post.xlsx (sheet: Tabelle1) as an abbreviation 
    for 4_E_T_IV. This is the only instance where E is used without E_T.
    
    Args:
        long_df: Long format DataFrame
        
    Returns:
        DataFrame with normalized column names
    """
    # Metadata columns are preserved as-is during normalization
    metadata_cols = identify_metadata_columns()
    
    columns_to_normalize = [col for col in long_df.columns if col not in metadata_cols]
    
    # Note: 4_E_IV → 4_E_T_IV renaming is now done during data extraction (see process_sheet_to_long_format)
    
    # Group columns by normalized names
    normalized_groups = {}
    
    for col in columns_to_normalize:
        col_str = str(col)
        
        # Remove pandas-added suffixes (.1, .2, .3, etc.)
        base_name = re.sub(r'\.\d+$', '', col_str)
        
        # Remove item numbers from coding columns
        if re.match(r'^\d+_', base_name):
            base_name = normalize_coding_column_name(base_name)
        
        if base_name not in normalized_groups:
            normalized_groups[base_name] = []
        normalized_groups[base_name].append(col)
    
    # Normalize columns: merge duplicates and clean names
    new_columns_created = {}
    columns_to_drop = []
    merged_categories = []
    
    for normalized_name, original_cols in normalized_groups.items():
        # Skip if column already has correct name
        if len(original_cols) == 1 and original_cols[0] == normalized_name:
            continue
        
        # Check for conflicts when merging multiple columns
        if len(original_cols) > 1:
            cols_data = long_df[original_cols]
            cols_data = cols_data.replace(list(MISSING_DATA_CODES), pd.NA)
            non_null_counts = cols_data.notna().sum(axis=1)
            rows_with_multiple = non_null_counts > 1
            if rows_with_multiple.any():
                conflict_idx = rows_with_multiple.idxmax()
                values = [long_df.at[conflict_idx, col] for col in original_cols]
                non_null_values = [v for v in values if pd.notna(v) and str(v).strip() not in MISSING_DATA_CODES]
                unique_values = set(str(v) for v in non_null_values)
                if len(unique_values) > 1:
                    print(f"\n✗ ERROR: Conflict detected during normalization!")
                    print(f"  Response ID: {long_df.at[conflict_idx, 'response_id']}")
                    print(f"  Column: {normalized_name}")
                    print(f"  Conflicting values: {list(non_null_values)}")
                    print(f"  Source columns: {list(original_cols)}")
                    raise ValueError(f"Data conflict in response {long_df.at[conflict_idx, 'response_id']} for column '{normalized_name}'")
        
        # Normalize: merge multiple columns or rename single column
        if len(original_cols) > 1:
            merged_data = long_df[original_cols].bfill(axis=1).iloc[:, 0]
            new_columns_created[normalized_name] = merged_data
            merged_categories.append(f"{original_cols} → {normalized_name}")
        else:
            new_columns_created[normalized_name] = long_df[original_cols[0]]
        
        columns_to_drop.extend(original_cols)
    
    # Drop original columns and add normalized versions
    if columns_to_drop:
        long_df = long_df.drop(columns=columns_to_drop)
        for col_name, col_data in new_columns_created.items():
            long_df[col_name] = col_data
        
        total_normalized = len(set(columns_to_drop))
        print(f"✓ Normalized {total_normalized} column(s)")
        
        if merged_categories:
            merged_categories.sort()
            print(f"\n  Merged {len(merged_categories)} duplicate categories:")
            for merge in merged_categories[:5]:  # Show first 5
                print(f"    {merge}")
            if len(merged_categories) > 5:
                print(f"    ... and {len(merged_categories) - 5} more")
        
        print("\n  ✓ No conflicts detected")
    else:
        print("✓ No columns needed normalization")
    
    return long_df


def clean_dataset(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes non-standard and empty columns from dataset.
    
    Args:
        long_df: Long format DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Remove non-standard columns
    non_standard_cols = ['Scaffold']
    cols_to_remove = [col for col in non_standard_cols if col in long_df.columns]
    if cols_to_remove:
        long_df = long_df.drop(columns=cols_to_remove)
        print(f"✓ Removed {len(cols_to_remove)} non-standard column(s): {', '.join(cols_to_remove)}")
    
    # Remove empty columns
    empty_cols = []
    for col in long_df.columns:
        col_data = long_df[col]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
        non_null_count = col_data.notna().sum()
        if non_null_count == 0:
            empty_cols.append(col)
    
    if empty_cols:
        print(f"✓ Removed {len(empty_cols)} empty column(s): {', '.join(empty_cols)}")
        long_df = long_df.drop(columns=empty_cols)
    else:
        print("✓ No empty columns found")
    
    return long_df


def sort_columns(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts columns: metadata first, then coding columns alphabetically.
    
    Args:
        long_df: Long format DataFrame
        
    Returns:
        DataFrame with sorted columns
    """
    metadata_cols = identify_metadata_columns()
    existing_metadata = [col for col in metadata_cols if col in long_df.columns]
    coding_cols = sorted([col for col in long_df.columns if col not in metadata_cols])
    
    long_df = long_df[existing_metadata + coding_cols]
    print(f"✓ Sorted {len(coding_cols)} coding columns alphabetically")
    
    return long_df
