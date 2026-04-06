#!/usr/bin/env python3
"""
Quick test script for RAG system.

Usage:
    python scripts/rag/test_rag.py
    python scripts/rag/test_rag.py --category 1_D_M --text "student shows no interest"
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_embeddings():
    """Test embedding model."""
    logger.info("Testing embedding model...")

    from src.rag import EmbeddingModel

    embedder = EmbeddingModel()
    logger.info(f"Model: {embedder.model_name}")
    logger.info(f"Dimension: {embedder.embedding_dim}")
    logger.info(f"Device: {embedder.device}")

    # Test embedding
    texts = [
        "Der Schüler zeigt kein Interesse am Unterricht",
        "Fehlendes Wecken von Interesse",
        "situationales Interesse schaffen",
    ]

    embeddings = embedder.embed(texts)
    logger.info(f"Embedded {len(texts)} texts, shape: {embeddings.shape}")

    # Test similarity
    similarities = embedder.similarity(texts[0], texts[1:])
    logger.info(f"Similarity to first text: {similarities}")

    logger.info("Embedding test passed!")
    return True


def test_vector_store():
    """Test vector store."""
    logger.info("Testing vector store...")

    from src.rag import EmbeddingModel, VectorStore

    embedder = EmbeddingModel()

    # Create store
    store = VectorStore(embedding_dim=embedder.embedding_dim)

    # Add documents
    texts = [
        "Der Schüler zeigt kein Interesse am Unterricht",
        "Fehlendes Wecken von Interesse",
        "situationales Interesse schaffen",
        "keine kognitive Aktivierung",
        "Wiederholung der letzten Unterrichtsstunde",
    ]

    metadata = [
        {"category": "1_D_M", "rating": 0},
        {"category": "1_D_M", "rating": 1},
        {"category": "1_D_M", "rating": 2},
        {"category": "1_D_M", "rating": 1},
        {"category": "1_D_M", "rating": 2},
    ]

    embeddings = embedder.embed_documents(texts)
    store.add_documents(texts, embeddings, metadata)

    logger.info(f"Added {len(store)} documents")

    # Test search
    query = "kein Interesse"
    query_emb = embedder.embed_query(query)
    results = store.search(query_emb, k=3)

    logger.info(f"Search results for '{query}':")
    for r in results:
        logger.info(f"  Score: {r['score']:.3f} | {r['text'][:50]}...")

    # Test category filter
    results = store.search_by_category(query_emb, category="1_D_M", k=2)
    logger.info(f"Category-filtered results: {len(results)}")

    logger.info("Vector store test passed!")
    return True


def test_retriever():
    """Test RAG retriever."""
    logger.info("Testing RAG retriever...")

    from src.rag import RAGRetriever

    retriever = RAGRetriever()

    # Index some test documents
    from src.rag.document_loader import Document

    documents = [
        Document(
            text="Der Schüler zeigt kein Interesse am Unterricht",
            metadata={"category": "1_D_M", "rating": 0},
        ),
        Document(
            text="Fehlendes Wecken von Interesse",
            metadata={"category": "1_D_M", "rating": 1},
        ),
        Document(
            text="situationales Interesse schaffen",
            metadata={"category": "1_D_M", "rating": 2},
        ),
    ]

    retriever.index_documents(documents)

    # Test search
    examples = retriever.get_similar_examples(
        query="Student ist nicht motiviert",
        category="1_D_M",
        k=2,
    )

    logger.info(f"Retrieved {len(examples)} similar examples:")
    for ex in examples:
        logger.info(f"  Rating {ex['rating']}: {ex['text'][:50]}...")

    # Test balanced retrieval
    balanced = retriever.get_balanced_examples(
        query="Student ist nicht motiviert",
        category="1_D_M",
        n_per_class=1,
    )

    logger.info(f"Balanced examples: {len(balanced)}")
    for ex in balanced:
        logger.info(f"  Rating {ex['rating']}: {ex['text'][:30]}...")

    logger.info("Retriever test passed!")
    return True


def test_full_classification(category: str, text: str, index_path: str):
    """Test full RAG classification."""
    logger.info("Testing full RAG classification...")

    from src.rag import RAGClassifier, RAGRetriever

    # Check if index exists
    index_dir = Path(index_path)
    if not index_dir.exists():
        logger.info("Creating test index...")

        retriever = RAGRetriever()

        # Try to load training data, fall back to examples from prompts.py
        train_path = Path("data/processed/train")
        if train_path.exists():
            retriever.index_training_data(str(train_path), categories=[category])
        else:
            from src.rag.document_loader import DocumentLoader
            loader = DocumentLoader()
            docs = loader.load_examples_from_prompts(category)
            if docs:
                retriever.index_documents(docs)
            else:
                logger.warning("No training data or examples found")
                return False

        retriever.save(str(index_dir))
    else:
        logger.info(f"Using existing index: {index_path}")

    # Create classifier
    logger.info("Creating RAG classifier...")
    classifier = RAGClassifier(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",  # Use small model for testing
        retriever_path=str(index_path),
    )

    # Classify
    logger.info(f"Classifying: '{text}'")
    result = classifier.classify(
        category=category,
        text=text,
        n_examples=3,
        use_balanced=True,
        return_examples=True,
    )

    logger.info(f"Result: {result['score']}")
    logger.info(f"Valid range: {result['valid_range']}")
    logger.info(f"Examples used: {result['n_examples_used']}")

    if result.get("examples"):
        logger.info("Retrieved examples:")
        for ex in result["examples"]:
            logger.info(f"  Rating {ex['rating']}: {ex['text'][:40]}...")

    # Cleanup
    classifier.unload()

    logger.info("Full classification test passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test RAG system")
    parser.add_argument("--category", default="1_D_M", help="Category to test")
    parser.add_argument(
        "--text",
        default="Der Schüler zeigt kein Interesse",
        help="Text to classify",
    )
    parser.add_argument(
        "--index-path",
        default="data/rag/test_index",
        help="Path to RAG index",
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip full classification test (requires LLM)",
    )

    args = parser.parse_args()

    all_passed = True

    # Run tests
    try:
        all_passed &= test_embeddings()
    except Exception as e:
        logger.error(f"Embedding test failed: {e}")
        all_passed = False

    try:
        all_passed &= test_vector_store()
    except Exception as e:
        logger.error(f"Vector store test failed: {e}")
        all_passed = False

    try:
        all_passed &= test_retriever()
    except Exception as e:
        logger.error(f"Retriever test failed: {e}")
        all_passed = False

    if not args.skip_classification:
        try:
            all_passed &= test_full_classification(
                category=args.category,
                text=args.text,
                index_path=args.index_path,
            )
        except Exception as e:
            logger.error(f"Full classification test failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    if all_passed:
        logger.info("=" * 50)
        logger.info("All tests passed!")
        logger.info("=" * 50)
    else:
        logger.error("Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
