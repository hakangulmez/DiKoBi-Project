#!/usr/bin/env python3
"""
Ingest documents into the RAG system.

This script indexes your training data and custom documents
for retrieval-augmented generation.

Usage:
    # Index training data
    python scripts/rag/ingest_documents.py --train-dir data/processed/train

    # Index specific categories
    python scripts/rag/ingest_documents.py --categories 1_D_M 1_D_kA

    # Index additional PDF documents
    python scripts/rag/ingest_documents.py --pdf docs/coding_manual.pdf

    # Index with custom embedding model
    python scripts/rag/ingest_documents.py --embedding-model multilingual-large

    # Force reindex
    python scripts/rag/ingest_documents.py --force
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag import RAGRetriever, DocumentLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest documents into RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic indexing of training data
    python ingest_documents.py

    # Index specific categories
    python ingest_documents.py --categories 1_D_M 3_D_F

    # Add PDF documents
    python ingest_documents.py --pdf manual.pdf --pdf guidelines.pdf

    # Use different embedding model
    python ingest_documents.py --embedding-model german-semantic
        """
    )

    parser.add_argument(
        "--train-dir",
        type=str,
        default="data/processed/train",
        help="Directory containing training CSV files"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/rag/index",
        help="Output directory for RAG index"
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Specific categories to index (default: all)"
    )

    parser.add_argument(
        "--pdf",
        action="append",
        default=[],
        help="PDF files to index (can be specified multiple times)"
    )

    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Text files to index (can be specified multiple times)"
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default="multilingual-small",
        choices=[
            "multilingual-small",
            "multilingual-large",
            "german-semantic",
            "e5-multilingual",
            "bge-m3",
        ],
        help="Embedding model to use"
    )

    parser.add_argument(
        "--index-type",
        type=str,
        default="flat",
        choices=["flat", "ivf", "hnsw"],
        help="FAISS index type"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindex even if index exists"
    )

    parser.add_argument(
        "--include-prompts-examples",
        action="store_true",
        default=True,
        help="Include examples from prompts.py"
    )

    parser.add_argument(
        "--no-prompts-examples",
        action="store_false",
        dest="include_prompts_examples",
        help="Don't include examples from prompts.py"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_path = Path(args.output)

    # Check if index exists
    if output_path.exists() and not args.force:
        logger.info(f"Index already exists at {output_path}")
        logger.info("Use --force to reindex")

        # Load and show stats
        try:
            retriever = RAGRetriever.load(str(output_path))
            stats = retriever.get_stats()
            logger.info(f"Current index stats: {stats}")
            return
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
            logger.info("Will create new index")

    # Initialize retriever
    logger.info(f"Initializing RAG retriever with {args.embedding_model} embeddings")
    retriever = RAGRetriever(
        embedding_model=args.embedding_model,
        index_type=args.index_type,
    )

    # Check if training directory exists
    train_path = Path(args.train_dir)
    if train_path.exists():
        logger.info(f"Indexing training data from {args.train_dir}")
        retriever.index_training_data(
            train_dir=args.train_dir,
            categories=args.categories,
            include_prompts_examples=args.include_prompts_examples,
        )
    else:
        logger.warning(f"Training directory not found: {args.train_dir}")
        if args.include_prompts_examples:
            logger.info("Indexing examples from prompts.py only")
            from src.rag.document_loader import DocumentLoader
            from src.classification.prompts import CATEGORIES

            loader = DocumentLoader()
            all_docs = []
            for category in CATEGORIES.keys():
                if args.categories and category not in args.categories:
                    continue
                docs = loader.load_examples_from_prompts(category)
                all_docs.extend(docs)

            if all_docs:
                retriever.index_documents(all_docs)

    # Index PDF files
    for pdf_path in args.pdf:
        pdf_file = Path(pdf_path)
        if pdf_file.exists():
            logger.info(f"Indexing PDF: {pdf_path}")
            retriever.index_pdf(pdf_path)
        else:
            logger.warning(f"PDF not found: {pdf_path}")

    # Index text files
    for text_path in args.text:
        text_file = Path(text_path)
        if text_file.exists():
            logger.info(f"Indexing text file: {text_path}")
            retriever.index_text_file(text_path)
        else:
            logger.warning(f"Text file not found: {text_path}")

    # Save index
    logger.info(f"Saving RAG index to {output_path}")
    retriever.save(str(output_path))

    # Print stats
    stats = retriever.get_stats()
    logger.info("=" * 50)
    logger.info("Indexing complete!")
    logger.info(f"Total documents: {stats['total_documents']}")
    logger.info(f"Embedding dimension: {stats['embedding_dim']}")
    logger.info(f"Index type: {stats['index_type']}")
    logger.info(f"Categories: {stats.get('categories', {})}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
