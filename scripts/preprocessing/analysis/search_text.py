"""
Helper script for debugging datasets.
Search for specific text in all Excel files in the Datasets folder.

Usage on Linux/macOS: python scripts/preprocessing/analysis/search_text.py
Usage on Windows: python scripts\preprocessing\analysis\search_text.py
"""

import pandas as pd
from pathlib import Path
import sys

# Add root directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config import RAW_DATA_DIR

def column_index_to_letter(index: int) -> str:
    """Convert 0-based column index to Excel column letter (A, B, ..., Z, AA, AB, ...)"""
    result = ""
    index += 1  # Convert to 1-based
    while index > 0:
        index -= 1
        result = chr(65 + (index % 26)) + result
        index //= 26
    return result

def search_in_datasets(search_text: str, folder_path: str = RAW_DATA_DIR):
    """
    Search for a specific text string in all Excel files.
    
    Args:
        search_text: The text to search for
        folder_path: Path to folder containing Excel files
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ Folder not found: {folder_path}")
        return
    
    # Find all Excel files
    excel_files = list(folder.glob('*.xlsx')) + list(folder.glob('*.xls'))
    
    if not excel_files:
        print(f"❌ No Excel files found in {folder_path}")
        return
    
    print("=" * 70)
    print(f"Searching for: '{search_text}'")
    print("=" * 70)
    print(f"\nSearching in {len(excel_files)} file(s)...\n")
    
    found_results = []
    
    for file in excel_files:
        print(f"📄 Checking {file.name}...")
        
        try:
            df = pd.read_excel(file)
            
            # Search in all columns
            for col_idx, col in enumerate(df.columns):
                for row_idx, value in enumerate(df[col]):
                    if pd.notna(value) and search_text in str(value):
                        found_results.append({
                            'file': file.name,
                            'row': row_idx + 2,  # +2 for Excel row (1-indexed, +1 for header)
                            'column': col,
                            'column_index': col_idx,
                            'column_letter': column_index_to_letter(col_idx),
                            'value': str(value)
                        })
                        print(f"  ✓ FOUND in row {row_idx + 2}, column '{col}'")
        
        except Exception as e:
            print(f"  ✗ Error reading {file.name}: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    if not found_results:
        print("❌ Text not found in any dataset.")
    else:
        print(f"✓ Found {len(found_results)} occurrence(s):\n")
        
        for result in found_results:
            print(f"📁 File: {result['file']}")
            print(f"   Row: {result['row']} (Excel)")
            print(f"   Column: {result['column']} ({result['column_letter']})")
            print(f"   Text preview: {result['value'][:200]}...")
            print()

if __name__ == "__main__":
    search_text = "5_W"  # Change this to the text you want to search for
    search_in_datasets(search_text)