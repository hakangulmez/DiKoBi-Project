# RAG System for DiKoBi

This module provides Retrieval-Augmented Generation (RAG) capabilities for the DiKoBi classification system. It enhances few-shot learning by dynamically retrieving relevant examples from your training data.

## Overview

The RAG system consists of:

- **EmbeddingModel**: Text embeddings using sentence-transformers (multilingual support)
- **VectorStore**: FAISS-based vector storage for efficient similarity search
- **DocumentLoader**: Load documents from CSV, PDF, and text files
- **RAGRetriever**: High-level semantic search interface
- **RAGClassifier**: RAG-enhanced classifier combining retrieval with LLM classification

## Quick Start

### 1. Install Dependencies

```bash
pip install sentence-transformers faiss-cpu pypdf
```

### 2. Index Your Training Data

```python
from src.rag import RAGRetriever

# Create retriever and index training data
retriever = RAGRetriever(embedding_model="multilingual-small")
retriever.index_training_data("data/processed/train")
retriever.save("data/rag/index")
```

Or use the CLI script:

```bash
python scripts/rag/ingest_documents.py --train-dir data/processed/train
```

### 3. Classify with RAG

```python
from src.rag import RAGClassifier

# Load classifier with RAG
classifier = RAGClassifier(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    retriever_path="data/rag/index"
)

# Classify with retrieved examples
result = classifier.classify(
    category="1_D_M",
    text="Der Schüler zeigt kein Interesse",
    n_examples=3,
    use_balanced=True
)

print(f"Score: {result['score']}")
```

## How It Works

1. **Indexing**: Training data is embedded using sentence-transformers and stored in a FAISS index
2. **Retrieval**: For each classification, similar examples are retrieved based on semantic similarity
3. **Prompt Building**: Retrieved examples are included in the LLM prompt as few-shot context
4. **Classification**: The LLM classifies the input using the dynamically retrieved examples

## Components

### EmbeddingModel

Supports multiple multilingual embedding models:

```python
from src.rag import EmbeddingModel

# Available models
embedder = EmbeddingModel(model_name="multilingual-small")  # Fast, good quality
embedder = EmbeddingModel(model_name="multilingual-large")  # Better quality
embedder = EmbeddingModel(model_name="german-semantic")     # German-optimized
embedder = EmbeddingModel(model_name="e5-multilingual")     # State-of-the-art
```

### VectorStore

FAISS-based vector storage with filtering:

```python
from src.rag import VectorStore

store = VectorStore(embedding_dim=384)
store.add_documents(texts, embeddings, metadata)
results = store.search(query_embedding, k=5)
results = store.search_by_category(query_embedding, category="1_D_M", k=5)
```

### RAGRetriever

High-level retrieval with category support:

```python
from src.rag import RAGRetriever

retriever = RAGRetriever()
retriever.index_training_data("data/processed/train")

# Get similar examples for classification
examples = retriever.get_similar_examples(
    query="student shows no interest",
    category="1_D_M",
    k=5
)

# Get balanced examples across rating classes
balanced = retriever.get_balanced_examples(
    query="student shows no interest",
    category="1_D_M",
    n_per_class=1
)
```

### RAGClassifier

Full RAG-enhanced classification:

```python
from src.rag import RAGClassifier

classifier = RAGClassifier(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    retriever_path="data/rag/index"
)

# Single classification
result = classifier.classify(
    category="1_D_M",
    text="...",
    n_examples=3,
    use_balanced=True,
    return_examples=True
)

# Batch classification
results = classifier.classify_batch(
    category="1_D_M",
    texts=["text1", "text2", ...],
    n_examples=3
)
```

## Adding Custom Documents

### Index PDF Documents

```python
retriever.index_pdf("docs/coding_manual.pdf")
```

Or via CLI:

```bash
python scripts/rag/ingest_documents.py --pdf docs/coding_manual.pdf
```

### Index Text Files

```python
retriever.index_text_file("docs/guidelines.txt")
```

### Index Custom Data

```python
from src.rag import DocumentLoader, RAGRetriever

loader = DocumentLoader()
documents = loader.load_documents([
    {"text": "Example text 1", "rating": 1, "category": "1_D_M"},
    {"text": "Example text 2", "rating": 2, "category": "1_D_M"},
])

retriever = RAGRetriever()
retriever.index_documents(documents)
```

## Comparison: RAG vs Static Few-Shot

| Aspect | Static Few-Shot | RAG Few-Shot |
|--------|-----------------|--------------|
| Examples | Same for all inputs | Dynamically selected |
| Relevance | May be irrelevant | Semantically similar |
| Scalability | Limited by prompt length | Uses large example pool |
| Adaptability | Fixed examples | Adapts to input |

## Best Practices

1. **Index all training data**: More examples = better retrieval
2. **Use balanced retrieval**: Ensures representation from all rating classes
3. **Include PDFs**: Index coding manuals for context
4. **Use multilingual models**: German text benefits from multilingual embeddings
5. **Test retrieval**: Use `retriever.get_similar_examples()` to verify quality

## Troubleshooting

### Memory Issues

```python
# Use smaller embedding model
embedder = EmbeddingModel(model_name="multilingual-small")

# Or use IVF index for large datasets
retriever = RAGRetriever(index_type="ivf")
```

### Slow Indexing

```python
# Use GPU if available
embedder = EmbeddingModel(device="cuda")
store = VectorStore(embedding_dim=384, use_gpu=True)
```

### Poor Retrieval Quality

1. Try different embedding models
2. Ensure training data quality
3. Adjust number of examples retrieved
4. Use balanced retrieval for classification tasks
