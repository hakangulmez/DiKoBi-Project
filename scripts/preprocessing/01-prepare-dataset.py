"""
DiKoBi Data Preparation Script
================================
Converts Excel data to long format (one row per free text response)
with associated coding ratings. Each response contains only its own codings,
avoiding the column explosion problem from traditional wide-to-long conversion.

The script:
- Reads all Excel files from the data/raw folder
- Extracts free text responses (columns starting with 'v_')
- Associates each response with its corresponding coding columns
- Normalizes column names by removing item numbers and merging duplicates
- Removes non-standard columns

Based on the DiKoBi coding manual, which assesses diagnostic competencies
of biology teachers using video-based simulations.

Usage on Linux/macOS: python scripts/preprocessing/01-prepare-dataset.py
Usage on Windows: python scripts/preprocessing/01-prepare-dataset.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.preprocessing.dataset_builder import (
    read_excel_files_to_long_format,
    normalize_dataset_columns,
    clean_dataset,
    sort_columns
)


def main(input_folder: str = RAW_DATA_DIR,
         output_file: str = PROCESSED_DATA_DIR / 'dikobi_long_format.csv'):
    """
    Orchestrates the complete data preparation pipeline for DiKoBi dataset.
    
    Converts Excel files directly to long format, normalizes column names, removes
    artifacts, and exports to CSV. Direct-to-long conversion prevents pandas from
    creating columns for all codings across all responses (column explosion).
    
    Args:
        input_folder: Folder containing source Excel files (default: data/raw/)
        output_file: Path to output CSV file (default: data/processed/dikobi_long_format.csv)
    """
    print("=" * 70)
    print("DiKoBi Data Preparation Pipeline")
    print("=" * 70)
    print()
    
    # Create output folder
    output_path = Path(output_file).parent
    output_path.mkdir(parents=True, exist_ok=True)
    
    # === Step 1: Read Excel files directly to long format ===
    print("[Step 1] Reading Excel files and converting to long format...")
    print("-" * 70)
    long_df = read_excel_files_to_long_format(input_folder)
    
    # === Step 2: Normalize coding column names and merge duplicates ===
    print("\n[Step 2] Normalizing coding column names...")
    print("-" * 70)
    long_df = normalize_dataset_columns(long_df)
    
    # === Step 3: Clean up columns ===
    print("\n[Step 3] Cleaning up data...")
    print("-" * 70)
    long_df = clean_dataset(long_df)
    
    # === Step 4: Sort columns and save ===
    print("\n[Step 4] Sorting columns and saving...")
    print("-" * 70)
    long_df = sort_columns(long_df)
    
    long_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✓ Saved to: {output_file}")
    
    # === Summary statistics ===
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total free text responses: {len(long_df)}")
    
    if 'participant_id' in long_df.columns:
        n_participants = long_df['participant_id'].nunique()
        print(f"Total unique participants: {n_participants}")
        print(f"Average responses per participant: {len(long_df) / n_participants:.1f}")
    
    print("\nColumn overview:")
    metadata_count = len([col for col in long_df.columns 
                         if col in ['response_id', 'participant_id', 'source_file', 
                                   'source_sheet', 'free_text_column', 'free_text_answer']])
    import re
    coding_count = len([col for col in long_df.columns if re.match(r'^\d+_', str(col))])
    print(f"  • ID/metadata columns: {metadata_count}")
    print(f"  • Coding columns: {coding_count}")
    
    return long_df


if __name__ == "__main__":
    try:
        prepared_data = main()
        print("\n✓ Data preparation completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
