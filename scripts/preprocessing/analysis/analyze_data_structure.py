"""
DiKoBi Data Quality Analyzer
==============================
Analyzes the dataset to identify data quality issues and category structure:
1. Mixed case numbers (responses coded for multiple video cases)
2. Mixed primary components (responses with multiple D/E_T/P/Dm codes)
3. Category distribution across cases and components
4. Files and sheets with data quality issues

Outputs:
- Console: Detailed analysis report with 10 sections
- CSV: data/processed/dikobi_category_structure.csv (enriched long-format with parsed components)

Usage on Linux/macOS: python scripts/preprocessing/analysis/analyze_data_structure.py
Usage on Windows: python scripts\preprocessing\analysis\analyze_data_structure.py
"""

import sys
from pathlib import Path
from typing import Union
import pandas as pd
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from config import PROCESSED_DATA_DIR


def extract_component_from_category(category: str) -> str:
    """
    Extracts the component (D/E_T/P/Dm/Va/W/Pha/Pv) from category name.
    
    Components represent the diagnostic competency being assessed.
    All components are equally important in the analysis.
    
    Note: The variant 4_E_IV (without _T) was merged into 4_E_T_IV during dataset creation,
    so all Explanation categories are now E_T.
    
    Category structure: {case}_{component}_{aspect}_{knowledge_type}
    
    Examples:
        '1_D_kA' -> 'D' (Description)
        '1_E_T_PCK' -> 'E_T' (Explanation-Theory)
        '4_E_T_IV' -> 'E_T' (Explanation-Theory, includes merged 4_E_IV cases)
        '2_P_PCK' -> 'P' (Prediction)
        '3_Dm_Quan' -> 'Dm' (Decision Making)
        '3_Pha' -> 'Pha' (TODO: not in manual?)
        '3_Pv_PCK' -> 'Pv' (TODO: not in manual?)
    """
    # Split by underscore to analyze parts
    parts = str(category).split('_')
    
    if len(parts) < 2:
        return 'UNKNOWN'
    
    # Check for E_T (two-part component)
    if len(parts) >= 3 and parts[1] == 'E' and parts[2] == 'T':
        return 'E_T'
    
    # Second part contains the component for single-part components
    component_part = parts[1]
    
    # Return the component (order matters: check Dm before D to avoid confusion)
    if component_part == 'Dm':
        return 'Dm'
    elif component_part == 'D':
        return 'D'
    elif component_part == 'P':
        return 'P'
    elif component_part == 'Va':
        return 'Va'
    elif component_part == 'W':
        return 'W'
    else:
        # Unknown component
        return component_part


def extract_case_number(category: str) -> int:
    """
    Extracts the case number from category name.
    
    Examples:
        '1DkA' -> 1
        '2DUSF' -> 2
        '3DmQuan' -> 3
    """
    match = re.match(r'^(\d+)', category)
    if match:
        return int(match.group(1))
    return None


def extract_aspect(category: str) -> str:
    """
    Extracts the specific aspect from category name.
    
    Examples:
        '1_D_kA' -> 'kA' (kognitive Aktivierung)
        '2_D_USF' -> 'USF' (Umgang mit Schülerfehlvorstellungen)
        '3_Dm_Quan' -> 'Quan' (Quantität)
        '1_E_T_IV' -> 'IV' (Interview - T is part of E_T component)
        '1_D_PCK' -> None (no aspect, just knowledge type)
        '3_Pha' -> None (no aspect, just component)
        '3_Pha_PCK' -> None (no aspect, just component + knowledge type)
        '3_Pha_F' -> 'F' (Pha component with F aspect)
        
    Note: 4_E_IV was merged into 4_E_T_IV during dataset creation.
    """
    # Remove case number prefix (e.g., '1_')
    clean = re.sub(r'^\d+_', '', category)
    
    # Remove component prefix (D_, E_, P_, Dm_, E_T_, Pha_, Pv_, W_, Va_)
    # E_T is special case (two parts), handle it first
    clean = re.sub(r'^E_T_', '', clean)
    clean = re.sub(r'^(D_|E_|P_|Dm_|Pha_|Pv_|W_|Va_)', '', clean)
    
    # Remove knowledge type suffix (_PCK or _PK)
    clean = re.sub(r'_(PCK|PK)$', '', clean)
    
    # If what remains is ONLY PCK/PK or a component name without aspect, return None
    if clean in ['PCK', 'PK', 'Pha', 'Pv', 'W', 'Va', 'D', 'Dm', 'P', 'E', ''] or not clean:
        return None
    
    return clean


def extract_knowledge_type(category: str) -> str:
    """
    Extracts knowledge type (PCK/PK) from category name.
    
    Examples:
        '1DPCK' -> 'PCK'
        '2P_PK' -> 'PK'
        '1DkA' -> None (not specified)
    """
    if 'PCK' in category:
        return 'PCK'
    elif 'PK' in category:
        return 'PK'
    return None


def analyze_data_structure(input_file: Union[str, Path] = PROCESSED_DATA_DIR / 'dikobi_long_format.csv'):
    """Analyzes the structure of the dataset to check case and component consistency."""
    
    print("=" * 80)
    print("DiKoBi Data Structure Analysis")
    print("=" * 80)
    print()
    
    # Load data
    df_wide = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df_wide)} rows from {input_file}")
    print(f"  Total columns: {len(df_wide.columns)}")
    print()
    
    # Identify category columns (those starting with a digit)
    metadata_cols = ['response_id', 'participant_id', 'source_file', 'source_sheet', 
                     'free_text_column', 'free_text_answer']
    category_cols = [col for col in df_wide.columns 
                     if col not in metadata_cols and re.match(r'^\d+_', str(col))]
    
    print(f"  Metadata columns: {len(metadata_cols)}")
    print(f"  Category columns: {len(category_cols)}")
    print()
    
    # Convert wide format to long format for analysis
    print("Converting to long format for analysis...")
    records = []
    
    for idx, row in df_wide.iterrows():
        response_id = row.get('response_id', idx + 1)
        
        for cat_col in category_cols:
            value = row[cat_col]
            # Only include non-null values
            if pd.notna(value) and str(value).strip() not in ['', '-66', '-99']:
                records.append({
                    'response_id': response_id,
                    'category': cat_col,
                    'rating': value
                })
    
    # Create long format dataframe
    df = pd.DataFrame(records)
    print(f"✓ Created long format with {len(df)} category ratings")
    print()
    
    # Extract structural information from category names
    # Category structure: {case}_{component}_{aspect}_{knowledge_type}
    df['case'] = df['category'].apply(extract_case_number)
    df['component'] = df['category'].apply(extract_component_from_category)
    df['aspect'] = df['category'].apply(extract_aspect)
    df['knowledge_type'] = df['category'].apply(extract_knowledge_type)
    
    # Analysis 1: Categories per row
    print("=" * 80)
    print("ANALYSIS 1: Categories per Response")
    print("=" * 80)
    
    response_groups = df.groupby('response_id')
    categories_per_response = response_groups['category'].count()
    components_per_response = response_groups['component'].nunique()
    cases_per_response = response_groups['case'].nunique()
    
    print(f"\nCategories per response:")
    print(f"  Min: {categories_per_response.min()}")
    print(f"  Max: {categories_per_response.max()}")
    print(f"  Mean: {categories_per_response.mean():.2f}")
    print(f"  Median: {categories_per_response.median():.0f}")
    
    print(f"\nComponents (D/E_T/P/Dm) per response:")
    print(components_per_response.value_counts().sort_index())
    
    print(f"\nCase numbers per response:")
    print(cases_per_response.value_counts().sort_index())
    
    # Analysis 2: Component distribution
    print("\n" + "=" * 80)
    print("ANALYSIS 2: Component Distribution")
    print("=" * 80)
    print(f"\n{df['component'].value_counts()}")
    
    # Analysis 3: Case distribution
    print("\n" + "=" * 80)
    print("ANALYSIS 3: Case Distribution")
    print("=" * 80)
    print(f"\n{df['case'].value_counts().sort_index()}")
    
    # Analysis 4: Knowledge type distribution
    print("\n" + "=" * 80)
    print("ANALYSIS 4: Knowledge Type Distribution")
    print("=" * 80)
    knowledge_dist = df['knowledge_type'].value_counts(dropna=False)
    print(f"\n{knowledge_dist}")
    
    # Analysis 5: Sample rows
    print("\n" + "=" * 80)
    print("ANALYSIS 5: Sample Rows (First 10 Categories)")
    print("=" * 80)
    print()
    
    sample_df = df[['response_id', 'category', 'case', 'component', 'aspect', 
                    'knowledge_type', 'rating']].head(10)
    print(sample_df.to_string(index=False))
    
    # Analysis 6: Unique aspects per case and component
    print("\n" + "=" * 80)
    print("ANALYSIS 6: Aspects per Case and Component")
    print("=" * 80)
    print()
    
    aspect_summary = df.groupby(['case', 'component'])['aspect'].apply(
        lambda x: list(x.unique())
    ).reset_index()
    
    for case in sorted(df['case'].unique()):
        case_data = aspect_summary[aspect_summary['case'] == case]
        print(f"\nCase {case}:")
        for _, row in case_data.iterrows():
            # Filter out None values before joining
            aspects = [str(a) for a in row['aspect'] if a is not None]
            aspects_str = ', '.join(aspects) if aspects else '(no aspect)'
            print(f"  {row['component']:8s}: {aspects_str}")
    
    # Analysis 7: Check if one response can have multiple ratings
    print("\n" + "=" * 80)
    print("ANALYSIS 7: Multiple Ratings per Response?")
    print("=" * 80)
    print()
    
    multi_rating_responses = response_groups.filter(lambda x: len(x) > 1)
    if len(multi_rating_responses) > 0:
        sample_multi = multi_rating_responses.groupby('response_id').first().head(3)
        print("✓ Yes, responses can have multiple categories")
        print(f"\nExample response with multiple categories:")
        print(f"Response ID: {sample_multi.index[0]}")
        resp_data = df[df['response_id'] == sample_multi.index[0]]
        print(resp_data[['category', 'case', 'component', 'aspect', 'knowledge_type', 'rating']].to_string(index=False))
    else:
        print("✗ No, each response has only one category/rating")
    
    # Analysis 8: Check if rows have only one case number
    print("\n" + "=" * 80)
    print("ANALYSIS 8: Case Number Consistency per Row")
    print("=" * 80)
    print()
    
    # Group by response_id and check if all categories have the same case number
    single_case_per_row = response_groups.apply(lambda x: x['case'].nunique() == 1, include_groups=False)
    
    print(f"Total responses: {len(single_case_per_row)}")
    print(f"Responses with only ONE case number: {single_case_per_row.sum()} ({single_case_per_row.sum()/len(single_case_per_row)*100:.1f}%)")
    print(f"Responses with MIXED case numbers: {(~single_case_per_row).sum()} ({(~single_case_per_row).sum()/len(single_case_per_row)*100:.1f}%)")
    
    # Show examples of mixed case numbers if any exist
    if (~single_case_per_row).any():
        print("\nExamples of responses with MIXED case numbers:")
        mixed_case_ids = single_case_per_row[~single_case_per_row].index[:5]
        for resp_id in mixed_case_ids:
            resp_data = df[df['response_id'] == resp_id]
            cases = resp_data['case'].unique()
            categories = resp_data['category'].tolist()
            print(f"\n  Response {resp_id}:")
            print(f"    Cases found: {sorted(cases)}")
            print(f"    Categories: {', '.join(categories[:10])}")
            if len(categories) > 10:
                print(f"    ... and {len(categories) - 10} more")
    else:
        print("\n✓ All responses contain only ONE case number (e.g., all starting with '1_' or all '2_', etc.)")
    
    # Analysis 9: Check if rows have only one component type or mixed
    print("\n" + "=" * 80)
    print("ANALYSIS 9: Component Type Consistency per Row")
    print("=" * 80)
    print()
    
    # Group by response_id and check if all categories have the same component
    single_component_per_row = response_groups.apply(lambda x: x['component'].nunique() == 1, include_groups=False)
    
    print(f"Total responses: {len(single_component_per_row)}")
    print(f"Responses with only ONE component type: {single_component_per_row.sum()} ({single_component_per_row.sum()/len(single_component_per_row)*100:.1f}%)")
    print(f"Responses with MIXED component types: {(~single_component_per_row).sum()} ({(~single_component_per_row).sum()/len(single_component_per_row)*100:.1f}%)")
    
    # Show distribution of component combinations
    component_combinations = response_groups['component'].apply(lambda x: '+'.join(sorted(x.unique())))
    print("\nComponent combinations found:")
    print(component_combinations.value_counts().head(20))
    
    # Show examples of mixed components if any exist
    if (~single_component_per_row).any():
        print("\nExamples of responses with MIXED component types:")
        mixed_dim_ids = single_component_per_row[~single_component_per_row].index[:5]
        for resp_id in mixed_dim_ids:
            resp_data = df[df['response_id'] == resp_id]
            components = resp_data['component'].unique()
            categories = resp_data[['category', 'component']].values.tolist()
            print(f"\n  Response {resp_id}:")
            print(f"    Components found: {', '.join(sorted(components))}")
            for cat, dim in categories[:8]:
                print(f"      {cat} ({dim})")
            if len(categories) > 8:
                print(f"      ... and {len(categories) - 8} more")
    else:
        print("\n✓ All responses contain only ONE component type per row")
    
    # Analysis 10: Summary of files with data quality issues
    print("\n" + "=" * 80)
    print("ANALYSIS 10: Files with Data Quality Issues")
    print("=" * 80)
    print()
    
    # Get files with mixed cases
    files_with_mixed_cases = set()
    for resp_id in single_case_per_row[~single_case_per_row].index:
        file = df_wide[df_wide['response_id'] == resp_id]['source_file'].iloc[0]
        files_with_mixed_cases.add(file)
    
    # Get files with mixed components
    files_with_mixed_dims = set()
    for resp_id in single_component_per_row[~single_component_per_row].index:
        file = df_wide[df_wide['response_id'] == resp_id]['source_file'].iloc[0]
        files_with_mixed_dims.add(file)
    
    all_files = df_wide['source_file'].unique()
    problem_files = files_with_mixed_cases | files_with_mixed_dims
    
    print(f"Total files in dataset: {len(all_files)}")
    print(f"Files with data quality issues: {len(problem_files)}")
    print(f"Clean files: {len(all_files) - len(problem_files)}")
    
    if problem_files:
        print("\nFiles WITH data quality issues:")
        for file in sorted(problem_files):
            problems = []
            if file in files_with_mixed_cases:
                problems.append('Mixed Cases')
            if file in files_with_mixed_dims:
                problems.append('Mixed Components')
            print(f"  ✗ {file:40s} - {' + '.join(problems)}")
    
    # Save enriched data with parsed category structure for further analysis
    output_file = PROCESSED_DATA_DIR / 'dikobi_category_structure.csv'
    output_cols = ['response_id', 'category', 'case', 'component', 'aspect', 'knowledge_type', 
                   'participant_id', 'source_file', 'source_sheet', 'rating']
    # Only save columns that exist in the dataframe
    cols_to_save = [col for col in output_cols if col in df.columns]
    df[cols_to_save].to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ Category structure dataset saved to: {output_file}")
    
    return df


if __name__ == "__main__":
    try:
        df = analyze_data_structure()
        print("\n✓ Analysis completed successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()