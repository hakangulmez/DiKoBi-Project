# DiKoBi - AI-Powered Teacher Response Classification

DiKoBi (Diagnosekompetenzen von Biologielehrkraeften) is a research project of the University of Munich (LMU) using video-based classroom simulations. Teachers in training write evaluations of teaching practice, and experts score these responses with the DiKoBi coding system.

## Overview

This repository provides a reproducible workflow to preprocess DiKoBi data, define category-specific prompts, benchmark models, and select a feasible large language model (LLM) prompting strategy for automated scoring, including RAG-enhanced few-shot retrieval.

**What it does:**
- 📝 Converts Excel files to structured CSV datasets (train/test splits)
- ✍️ Defines category-specific prompts aligned to the DiKoBi manual
- 🧪 Benchmarks 13 LLMs
- 🎯 Compares zero-shot, few-shot example counts, RAG templates, and output formats (text/JSON)
- 📊 Selects best configurations by median QWK across categories
- 📈 Tracks experiments with per-category histories and summaries

## Quick Start

**→ Open [`diKobi_script.ipynb`](diKobi_script.ipynb)** for the complete guided workflow (preprocessing, prompt definition, model selection, and evaluation).

---
## DiKoBi Categories (26 Total)

The system classifies across 26 categories covering 6 lesson topics. Each category uses specific criteria from the [DiKoBi Coding Manual](https://epub.ub.uni-muenchen.de/77972/1/MCLSReports_Kodiermanual.pdf).

**Category Structure:** `{case}_{component}_{aspect}_{knowledge_type}`

- **Case (1-6):** Video case number
- **Component:** D=Description, E_T=Explanation-Theory, P=Prediction, Dm=Decision Making, etc.
- **Aspect:** Specific focus (M=Motivation, kA=Cognitive Activation, F=Fachsprache, etc.)
- **Knowledge Type:** PCK (Pedagogical Content Knowledge) or PK (Pedagogical Knowledge)

**Examples:**
- `1_D_M` → Case 1, Description, Motivation aspect
- `3_E_T_IV` → Case 3, Explanation, Interview/Evaluation
- `5_Dm_Ma` → Case 5, Remember, Model Application

Scores range from 0-2 or 0-3 depending on category.

---
## System Requirements

**Minimum:**
- Python 3.11+
- 8GB RAM

**Recommended:**
- Python 3.12+
- 16GB RAM
- GPU (NVIDIA 8GB+ or Apple Silicon) for faster processing

**Hardware Support:**
- ✅ **NVIDIA GPUs** - Fastest with CUDA acceleration
- ✅ **Apple Silicon** (M1/M2/M3/M4) - Fast with Metal acceleration
- ✅ **CPU-only** - Works on any computer (slower)

---
## Workflow and Experimental Design

1. **Data preprocessing**
   - Excel ingestion, validation, normalization, and train/test splits

2. **Define category-specific prompts** (26 categories)
   - Prompts aligned to the DiKoBi coding manual

3. **Model selection** (single category: 1_D_M)
   - Test 13 compatible models
   - Evaluate text and JSON output formats
   - Empirical token limit testing to minimize invalid outputs
   - **Result:** Qwen2.5-7B-Instruct outperformed others (Text format 5 tokens, JSON format 15 tokens)

| Model | Size | VRAM | Status | Notes |
|-------|------|------|--------|-------|
| Qwen3 0.6B | ~1GB | Any | Excluded | Bad performance in evaluation |
| Qwen2.5 0.5B | ~1GB | Any | Excluded | Bad performance in evaluation |
| Qwen2.5 1.5B | ~3GB | 8GB+ | Included | 2nd best overall |
| Qwen2.5 3B | ~6GB | 12GB+ | Excluded | Bad performance in evaluation |
| Qwen3 4B | ~8GB | 12GB+ | Included | 3rd best overall |
| Qwen2.5 7B | ~14GB | 16GB+ | Included | ⭐ Best overall performance |
| Qwen3 8B | ~16GB | 16GB+ | Excluded | Bad performance in evaluation |
| Apriel 5B | ~10GB | 16GB+ | Excluded | Compatibility issues (could not load) |
| OLMo 2 7B | ~14GB | 16GB+ | Excluded | Bad performance in evaluation |
| RNJ-1 8B | ~16GB | 16GB+ | Excluded | Many invalid answers |
| Nemotron 3 Nano 30B | ~60GB | 40GB+ | Excluded | Too large for available hardware |
| DeepSeek R1 Qwen3 8B | ~16GB | 16GB+ | Excluded | Bad performance, invalid answers |
| Llama 3.2 3B | ~6GB | 8GB+ | Excluded | Bad performance in evaluation |

4. **Few-shot optimization** (1_D_M with Qwen2.5-7B-Instruct)
   - Test 1, 2, 3 examples per rating class
   - **Result:** 1 example per rating achieved best QWK

5. **Comprehensive evaluation** (Qwen2.5-7B-Instruct across all 26 categories)
   - Templates: zero-shot, few-shot, RAG similarity, RAG balanced
   - Output formats: text, JSON
   - **Total:** 8 configurations per category (4 templates × 2 formats)

| Template | Output Format | Description |
|----------|---------------|-------------|
| `zero_shot` | text | No examples, simple text output |
| `zero_shot` | json | No examples, structured JSON output |
| `few_shot` | text | Static curated examples, text output |
| `few_shot` | json | Static curated examples, JSON output |
| `rag_similarity` | text | Dynamic RAG examples (semantic), text output |
| `rag_similarity` | json | Dynamic RAG examples (semantic), JSON output |
| `rag_balanced` | text | Dynamic RAG examples (balanced), text output |
| `rag_balanced` | json | Dynamic RAG examples (balanced), JSON output |

6. **Select best configuration**
   - Choose the configuration with the highest median QWK over all categories

7. **Production classification**
   - Apply the optimal configuration to new data

All experiments run from the notebook with automatic tracking and visualization.

---
## Technical Details
### Deterministic Testing
All experiments use deterministic settings to ensure reproducible results:

```python
# LLM Deterministic Settings
temperature = 0.0          # Deterministic output (no randomness)
top_p = 1.0                # No nucleus sampling
do_sample = False          # Greedy decoding only
random_seed = 42           # Fixed seed for reproducibility

# Model loading settings
use_8bit = True            # Reduce VRAM (NVIDIA + bitsandbytes)
```
**Where to change settings:** All development configurations (model, template, max samples, quantization, token limits) are set in Part 5 of the notebook.

### Evaluation Metrics
- **Primary:** QWK (Quadratic Weighted Kappa)
- **Secondary:** Accuracy, MSE
- Metrics are reported per category and summarized across categories (median QWK used for final configuration selection)

### Experiment Tracking
Per-category JSON files in `results/experiment_history/<category>.json`:
- Chronological list of all experiments
- Metrics, configurations, timestamps
- Easy comparison across iterations
- Duplicate detection by prompt hash

Additionally, each experiment creates a timestamped folder in `data/experiments/`:
- **predictions.csv** - Individual predictions with ground truth
- **experiment.json** - Full metadata (model, params, metrics)
- **bundle.json** - Combined experiment + predictions

### Visualization
Interactive exploration in notebook Part 12

### RAG System
The RAG system enhances few-shot classification by retrieving semantically similar, labeled examples from the training data for each new response and injecting them into the prompt at runtime.

**How it works:**
- **Retrieval:** embed the new response and search for similar labeled examples in the training set.
- **Contextual scoring:** insert retrieved examples as few-shot context to guide the model's rating decision.
- **Dynamic prompting:** examples change per response, matching the current text instead of using fixed examples.

**Pipeline:**
1. Text is embedded with `sentence-transformers`.
2. FAISS finds the most similar examples from the training data.
3. Retrieved examples are injected into the prompt as few-shot examples.
4. The LLM classifies using these contextually relevant examples.

**RAG components:**

| Component | Technology / Setting |
|----------|-----------------------|
| Embedding Model | sentence-transformers (multilingual-small) |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Retrieval Strategy | Similarity-based and class-balanced |
| Few-Shot Examples | n = 3 examples per classification |
| Index Content | Labeled training data + curated prompt examples (prompts.py) |

---
## Project Structure
```
diKobi/
├── README.md                  # Overview (this file)
├── diKobi_script.ipynb        # ⭐ START HERE - Complete guided notebook
├── config.py                  # Development settings
├── requirements.txt           # Python dependencies
│
├── data/
│   ├── raw/                   # Original Excel files (git-ignored)
│   ├── processed/             # Generated CSV files (git-ignored)
│   │   ├── dikobi_long_format.csv  # Combined dataset
│   │   ├── train/             # Training data (80%)
│   │   └── test/              # Test data (20%)
│   ├── experiments/           # Per-run artifacts (git-ignored)
│   │   └── experiment_<timestamp>__<category>__<exp_id>/
│   │       ├── predictions.csv
│   │       ├── experiment.json
│   │       └── bundle.json
│   └── rag/                   # RAG index and embeddings (git-ignored)
│       └── index/
│
├── results/                   # Experiment results (git-tracked)
│   ├── experiment_history/    # Per-category JSON history -> main results folder
│   ├── phase1_template_comparison/  # Phase 1 summaries
│   └── phase2_batch_all_categories/  # Phase 2 summaries
│
├── src/                       # Core library modules
│   ├── classification/        # LLM classification
│   │   ├── classifier.py         # Main TextClassifier class
│   │   └── prompts.py            # Category definitions & templates
│   ├── evaluation/            # Metrics & testing
│   │   ├── metrics.py            # QWK, accuracy, MSE, etc.
│   │   ├── compare_templates.py  # Template comparison
│   │   ├── compare_models.py     # Model comparison
│   │   ├── rag_evaluator.py      # RAG evaluation
│   │   └── batch_rag_evaluator.py # Batch RAG evaluation
│   ├── models/                # Model management
│   │   ├── registry.py           # Available models & compatibility
│   │   └── loader.py             # Model loading utilities
│   ├── preprocessing/         # Data preparation modules
│   │   ├── dataset_builder.py    # Dataset creation functions
│   │   ├── category_splitter.py  # Category extraction
│   │   ├── column_utils.py       # Column identification utilities
│   │   ├── validators.py         # Validation functions
│   │   └── standards.py          # DiKobi coding standards
│   ├── rag/                   # RAG system
│   │   ├── document_loader.py    # PDF loading
│   │   ├── embeddings.py         # Semantic embeddings
│   │   ├── rag_classifier.py     # RAG-based classification
│   │   ├── retriever.py          # Vector retrieval
│   │   └── vector_store.py       # FAISS vector store
│   ├── visualization/         # Results visualization
│   │   └── results_plotter.py   # Plotting utilities
│   └── utils/                 # Helper functions
│       ├── device.py             # GPU/CPU detection
│       └── experiment_tracker.py # Unified experiment tracking
│
├── scripts/                   # Executable scripts
│   ├── preprocessing/         # Data preprocessing (Excel → CSV)
│   │   ├── 00-preprocess-all.py     # Run all preprocessing steps
│   │   ├── 01-prepare-dataset.py    # Extract & combine
│   │   ├── 02-split-categories.py   # Split by category
│   │   ├── 03-train-test-split.py   # Create train/test splits
│   │   └── analysis/                # Data validation tools
│   │       ├── validate_extraction.py  # Verify completeness
│   │       ├── analyze_data_structure.py  # Check quality
│   │       └── search_text.py          # Search in Excel files
│   └── rag/                   # RAG system scripts
│       ├── ingest_documents.py      # Index training data
│       └── test_rag.py              # RAG testing
│
└── RAGtests/           # RAG system experiments - deprecated testing folder
```

---

**Ready to start? Open `diKobi_script.ipynb` and follow along! 🚀**