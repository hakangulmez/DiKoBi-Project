"""
Preprocessing modules for DiKobi dataset preparation.
"""

from .column_utils import (
    find_free_text_columns,
    get_coding_columns_for_text,
    normalize_coding_column_name,
    identify_metadata_columns,
    identify_category_columns,
)

from .dataset_builder import (
    clean_coding_value,
    read_excel_files_to_long_format,
    process_sheet_to_long_format,
    normalize_dataset_columns,
    clean_dataset,
    sort_columns,
)

from .category_splitter import (
    extract_category_dataset,
    create_all_category_datasets,
    get_category_statistics,
)

from .standards import MAX_VALUES

from .validators import (
    extract_all_codings_from_excel,
    analyze_processed_csv,
    validate_column_completeness,
    validate_normalization_consistency,
    validate_response_codings,
    check_missing_columns_details,
)

__all__ = [
    # Column utilities
    'find_free_text_columns',
    'get_coding_columns_for_text',
    'normalize_coding_column_name',
    'identify_metadata_columns',
    'identify_category_columns',
    
    # Dataset building
    'clean_coding_value',
    'read_excel_files_to_long_format',
    'process_sheet_to_long_format',
    'normalize_dataset_columns',
    'clean_dataset',
    'sort_columns',
    
    # Category splitting
    'extract_category_dataset',
    'create_all_category_datasets',
    'get_category_statistics',
    
    # Standards
    'MAX_VALUES',
    
    # Validation
    'extract_all_codings_from_excel',
    'analyze_processed_csv',
    'validate_column_completeness',
    'validate_normalization_consistency',
    'validate_response_codings',
    'check_missing_columns_details',
]
