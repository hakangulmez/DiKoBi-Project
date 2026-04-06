"""
Coding Extraction Validation Script
====================================
Validates that all codings from Excel files were correctly extracted.

Usage on Linux/macOS: python scripts/preprocessing/analysis/validate_extraction.py
Usage on Windows: python scripts/preprocessing/analysis/validate_extraction.py
"""

import sys
from pathlib import Path
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.preprocessing.validators import (
    extract_all_codings_from_excel,
    analyze_processed_csv,
    validate_column_completeness,
    validate_normalization_consistency,
    validate_response_codings,
    check_missing_columns_details
)


def print_validation_report(excel_data: dict, csv_data: dict, 
                           completeness: dict, normalization: dict,
                           response_codings: dict):
    """Prints comprehensive validation report."""
    print("=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)
    print()
    
    all_passed = completeness['pass'] and response_codings['pass']
    
    if all_passed:
        print("✓ ALL CHECKS PASSED - All codings correctly extracted!")
        print()
    else:
        print("⚠ VALIDATION RESULTS")
        print()
    
    # 1. Column Completeness
    print("-" * 70)
    print("1. Column Completeness Check")
    print("-" * 70)
    
    if completeness['pass']:
        print(f"✓ All {len(excel_data['columns'])} coding columns correctly extracted")
    else:
        if completeness['missing_in_csv']:
            # Detailed analysis of missing columns
            missing_details = check_missing_columns_details(
                completeness['missing_in_csv'], 
                RAW_DATA_DIR
            )
            
            # Categorize missing columns
            empty_cols = []
            orphaned_cols = []
            problematic_cols = []
            
            for col, details in missing_details.items():
                if details['total_values'] == 0:
                    empty_cols.append(col)
                elif not details['free_text_has_data']:
                    free_text = list(set(details['associated_free_text']))[0] if details['associated_free_text'] else 'unknown'
                    orphaned_cols.append((col, details['total_values'], free_text))
                else:
                    problematic_cols.append((col, details))
            
            # Only report problematic columns
            if problematic_cols:
                print(f"✗ PROBLEM: {len(problematic_cols)} columns with data were NOT extracted:\n")
                for col, details in problematic_cols:
                    print(f"  • {col}: {details['total_values']} codings")
                    free_text = list(set(details['associated_free_text']))[0] if details['associated_free_text'] else 'unknown'
                    print(f"    Free text: {free_text} (HAS data)")
                    print(f"    Locations: {', '.join(details['locations'][:3])}")
                print()
            else:
                # All missing columns are justified
                print(f"✓ All {len(completeness['missing_in_csv'])} missing columns are correctly excluded:")
                print(f"  • {len(empty_cols)} completely empty (no data)")
                if orphaned_cols:
                    print(f"  • {len(orphaned_cols)} orphaned (associated free text is empty)")
        
        if completeness['extra_in_csv']:
            print(f"\n⚠ {len(completeness['extra_in_csv'])} unexpected columns in CSV")
    
    print()
    
    # 2. Normalization
    print("-" * 70)
    print("2. Normalization Check")
    print("-" * 70)
    
    if normalization['merge_count'] > 0:
        print(f"✓ {normalization['merge_count']} columns merged")
    else:
        print("✓ No merging needed")
    
    print()
    
    # 3. Response Codings
    print("-" * 70)
    print("3. Response Codings Check")
    print("-" * 70)
    
    if response_codings['pass']:
        print(f"✓ All {response_codings['total_responses']:,} responses have codings")
    else:
        print(f"✗ {response_codings['responses_without_coding']} responses have NO codings")
    
    print()
    
    # 4. Summary
    print("-" * 70)
    print("4. Summary")
    print("-" * 70)
    print(f"Coding columns extracted: {len(csv_data['columns'])} / {len(excel_data['columns'])}")
    print(f"Responses in CSV: {csv_data['response_count']:,}")
    print(f"Responses with codings: {response_codings['responses_with_coding']:,}")
    
    print()
    print("=" * 70)
    print()
    
    # Check if all missing columns are justified
    all_missing_justified = True
    if not completeness['pass'] and len(completeness['missing_in_csv']) > 0:
        missing_details = check_missing_columns_details(
            completeness['missing_in_csv'],
            RAW_DATA_DIR
        )
        all_missing_justified = all(
            details['total_values'] == 0 or not details['free_text_has_data']
            for details in missing_details.values()
        )
    
    if all_missing_justified and response_codings['pass']:
        print("✅ VALIDATION PASSED: All codings correctly extracted")
        print("\n  No data loss - all exclusions justified")
        print("  Status: PRODUCTION READY 🎉")
    else:
        print("❌ VALIDATION FAILED: Issues found")
        print("\n  Review problems listed above")
    
    print("=" * 70)


def main():
    """Main validation function."""
    print("=" * 70)
    print("DiKoBi Coding Extraction Validation")
    print("=" * 70)
    print()
    
    try:
        print("[Step 1] Extracting codings from Excel files...")
        print("-" * 70)
        excel_data = extract_all_codings_from_excel(RAW_DATA_DIR)
        
        print("[Step 2] Analyzing processed CSV...")
        print("-" * 70)
        csv_data = analyze_processed_csv(PROCESSED_DATA_DIR / 'dikobi_long_format.csv')
        
        print("[Step 3] Running validation checks...")
        print("-" * 70)
        print()
        
        completeness = validate_column_completeness(excel_data, csv_data)
        normalization = validate_normalization_consistency(excel_data)
        response_codings = validate_response_codings(csv_data)
        
        print_validation_report(excel_data, csv_data, completeness, 
                               normalization, response_codings)
        
        print("✓ Validation complete!")
        
    except Exception as e:
        print(f"\n✗ Error during validation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
