"""
Batch RAG evaluation across multiple categories.

This module provides functions for running RAG evaluation across all categories,
testing both output formats (text and JSON) with appropriate token limits.
It reuses the core evaluation logic from rag_evaluator.py.
"""

from pathlib import Path
from typing import List, Optional
import time

from src.rag import RAGClassifier
from src.models import get_model_config
from src.evaluation.rag_evaluator import evaluate_rag_single_category


def evaluate_rag_all_categories(
    model_name: str,
    rag_index_path: Path,
    test_dir: Path,
    text_max_tokens: int = 5,
    json_max_tokens: int = 15,
    max_samples: Optional[int] = None,
    use_8bit: bool = True,
    n_examples: int = 3,
    batch_size: int = 2,
    temperature: float = 0.0,
    categories: Optional[List[str]] = None,
) -> None:
    """
    Run RAG evaluation across all categories with both text and JSON formats.
    
    This function:
    1. Discovers all categories in test_dir
    2. Tests both text and JSON output formats
    3. For each format, evaluates all categories
    4. Results are saved to ExperimentTracker automatically
    
    Args:
        model_name: Model identifier (e.g., "qwen3-4b")
        rag_index_path: Path to RAG index
        test_dir: Directory containing test CSV files
        text_max_tokens: Max tokens for text format
        json_max_tokens: Max tokens for JSON format
        max_samples: Maximum test samples per category (None = all)
        use_8bit: Use 8-bit quantization
        n_examples: Number of RAG examples to retrieve
        batch_size: Batch size for classification
        temperature: Sampling temperature
        categories: Specific categories to test (None = all)
    """
    print("Starting RAG Batch Evaluation (ALL categories)\n")
    
    # Check prerequisites
    if not rag_index_path.exists():
        raise FileNotFoundError(
            f"RAG index not found: {rag_index_path}. "
            "Please create the RAG index first."
        )
    if not test_dir.exists():
        raise FileNotFoundError(f"Test data not found: {test_dir}")
    
    # Discover categories
    if categories is None:
        categories = sorted([p.stem for p in test_dir.glob("*.csv")])
    
    if not categories:
        raise FileNotFoundError(f"No test CSVs found in: {test_dir}")
    
    # Get HuggingFace ID from model configuration
    model_cfg = get_model_config(model_name)
    model_hf_id = model_cfg.hf_id
    
    # Test both text and json formats
    output_formats = [
        ("text", text_max_tokens),
        ("json", json_max_tokens),
    ]
    
    print(f"Found {len(categories)} categories in {test_dir}")
    print("Categories:", ", ".join(categories[:5]), 
          f"... (+{len(categories)-5} more)" if len(categories) > 5 else "")
    print()
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"HF ID:        {model_hf_id}")
    print(f"Quantization: {'8-bit' if use_8bit else 'None'}")
    print(f"Test samples: {max_samples or 'Full test set'}")
    print(f"RAG examples: {n_examples}")
    print(f"Batch size:   {batch_size}")
    print(f"Temperature:  {temperature}")
    print(f"Output modes: TEXT and JSON (both will be tested)")
    print("=" * 80)
    
    try:
        # Loop through both formats (text and json)
        for fmt_idx, (output_format, max_tokens) in enumerate(output_formats, 1):
            print("\n" + "#" * 80)
            print(f"# FORMAT {fmt_idx}/{len(output_formats)}: {output_format.upper()}")
            print("#" * 80)
            print(f"HF ID:       {model_hf_id}")
            print(f"Max tokens:  {max_tokens}")
            print(f"Output:      {output_format}")
            print()
            
            # PHASE 1: Check which experiments need to run (no model loading yet!)
            print("Checking experiment cache...")
            experiments_to_run = []
            
            from src.utils import ExperimentTracker
            
            for category in categories:
                tracker = ExperimentTracker(category)
                
                # Check both RAG methods
                for method_name, template_key, use_balanced in [
                    ("RAG Balanced", "rag_balanced", True),
                    ("RAG Similarity", "rag_similarity", False),
                ]:
                    check_result = tracker.check_duplicate(
                        model=model_hf_id,
                        prompt_template=template_key,
                        max_samples=max_samples,
                        use_8bit=use_8bit,
                        max_new_tokens=max_tokens,
                        output_format=output_format,  # Pass the output format explicitly!
                        verbose=True,  # Show details to debug cache misses
                    )
                    
                    if check_result['cached']:
                        print(f"✓ [{category}] {method_name}: CACHED")
                    else:
                        experiments_to_run.append(category)
                        print(f"○ [{category}] {method_name}: NEEDS RUN")
                        break  # Only need to know if category needs ANY run
            
            # Remove duplicates from experiments_to_run
            experiments_to_run = list(dict.fromkeys(experiments_to_run))
            
            print(f"\nCache scan: {len(categories) - len(experiments_to_run)} cached, {len(experiments_to_run)} to run\n")
            
            # PHASE 2: Only load classifier if there's work to do
            if not experiments_to_run:
                print(f"🎉 All experiments for {output_format} format already cached! Skipping model load.\n")
                continue
            
            # Load RAG classifier for this format
            print(f"Loading RAG classifier ({output_format})...")
            rag_classifier = RAGClassifier(
                model_name=model_hf_id,
                retriever_path=str(rag_index_path),
                use_8bit=use_8bit,
                max_new_tokens=max_tokens,
                temperature=temperature,
                batch_size=batch_size,
            )
            print("✓ Classifier loaded\n")
            
            # Process only categories that need evaluation
            for ci, category in enumerate(experiments_to_run, 1):
                print("=" * 80)
                print(f"[{ci}/{len(experiments_to_run)}] Category: {category} ({output_format})")
                
                try:
                    # Use single-category evaluator
                    evaluate_rag_single_category(
                        category=category,
                        model_hf_id=model_hf_id,
                        rag_index_path=rag_index_path,
                        test_dir=test_dir,
                        max_samples=max_samples,
                        use_8bit=use_8bit,
                        max_new_tokens=max_tokens,
                        n_examples=n_examples,
                        batch_size=batch_size,
                        temperature=temperature,
                        output_format=output_format,
                        verbose=False,  # Suppress detailed output for batch
                    )
                    
                    print(f"  ✓ Category {category} completed")
                    
                except Exception as e:
                    print(f"  ❌ Category {category} FAILED: {e}")
                
                # Clear cache after each category to reduce memory growth
                rag_classifier.clear_cache()
            
            # Unload classifier after this format is done
            rag_classifier.unload()
            print(f"\n✓ Classifier unloaded ({output_format})\n")
        
        print("\n" + "=" * 80)
        print("RAG BATCH EVALUATION COMPLETED!")
        print("=" * 80)
        print("Results saved to: data/experiments/")
        print("View all results using ExperimentTracker.show_history()")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        try:
            rag_classifier.unload()
        except:
            pass
        raise
    except Exception as e:
        print(f"\n❌ Batch evaluation failed: {e}")
        try:
            rag_classifier.unload()
        except:
            pass
        raise


def evaluate_rag_categories_list(
    model_name: str,
    categories: List[str],
    rag_index_path: Path,
    test_dir: Path,
    max_tokens: int = 15,
    max_samples: Optional[int] = None,
    use_8bit: bool = True,
    n_examples: int = 3,
    batch_size: int = 2,
    temperature: float = 0.0,
    use_json: bool = False,
) -> None:
    """
    Evaluate specific categories with a single format configuration.
    
    Useful for targeted testing or when you only need one output format.
    
    Args:
        model_name: Model identifier (e.g., "qwen3-4b")
        categories: List of categories to evaluate
        rag_index_path: Path to RAG index
        test_dir: Directory containing test CSV files
        max_tokens: Maximum tokens for generation
        max_samples: Maximum test samples per category (None = all)
        use_8bit: Use 8-bit quantization
        n_examples: Number of RAG examples to retrieve
        batch_size: Batch size for classification
        temperature: Sampling temperature
        use_json: Use JSON output format
    """
    # Get HuggingFace ID from model configuration
    model_cfg = get_model_config(model_name)
    model_hf_id = model_cfg.hf_id
    
    print("=" * 80)
    print(f"RAG EVALUATION: {len(categories)} categories")
    print("=" * 80)
    print(f"Model:     {model_hf_id}")
    print(f"Format:    {'JSON' if use_json else 'TEXT'}")
    print(f"Max tokens: {max_tokens}")
    print("=" * 80)
    
    # Pre-load classifier
    print("Loading RAG classifier...")
    rag_classifier = RAGClassifier(
        model_name=model_hf_id,
        retriever_path=str(rag_index_path),
        use_8bit=use_8bit,
        max_new_tokens=max_tokens,
        temperature=temperature,
        batch_size=batch_size,
    )
    print("✓ Classifier loaded\n")
    
    try:
        for ci, category in enumerate(categories, 1):
            print(f"\n[{ci}/{len(categories)}] {category}")
            print("-" * 80)
            
            try:
                evaluate_rag_single_category(
                    category=category,
                    model_hf_id=model_hf_id,
                    rag_index_path=rag_index_path,
                    test_dir=test_dir,
                    max_samples=max_samples,
                    use_8bit=use_8bit,
                    max_new_tokens=max_tokens,
                    n_examples=n_examples,
                    batch_size=batch_size,
                    temperature=temperature,
                    use_json=use_json,
                    verbose=False,
                )
                print(f"  ✓ Completed")
                
            except Exception as e:
                print(f"  ❌ FAILED: {e}")
            
            # Clear cache
            rag_classifier.clear_cache()
    
    finally:
        rag_classifier.unload()
        print("\n✓ Classifier unloaded")
