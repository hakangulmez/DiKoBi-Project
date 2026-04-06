"""
RAG Retriever for semantic search and example retrieval.
High-level interface combining embeddings and vector store.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .document_loader import Document, DocumentLoader

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    High-level RAG retriever combining embedding and vector search.

    Provides semantic search over indexed documents, with special
    support for DiKoBi category-based retrieval.

    Example:
        # Create and index
        retriever = RAGRetriever()
        retriever.index_training_data("data/processed/train")
        retriever.save("data/rag/index")

        # Load and search
        retriever = RAGRetriever.load("data/rag/index")
        examples = retriever.get_similar_examples(
            query="student shows no interest",
            category="1_D_M",
            k=5
        )
    """

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        device: Optional[str] = None,
        index_type: str = "flat",
    ):
        """
        Initialize RAG retriever.

        Args:
            embedding_model: Embedding model name. None for default.
            device: Device for embeddings ('cuda', 'mps', 'cpu').
            index_type: FAISS index type ('flat', 'ivf', 'hnsw').
        """
        self.embedding_model_name = embedding_model
        self.device = device
        self.index_type = index_type

        # Initialize embedding model
        self.embedder = EmbeddingModel(
            model_name=embedding_model,
            device=device,
        )

        # Initialize vector store
        self.vector_store = VectorStore(
            embedding_dim=self.embedder.embedding_dim,
            index_type=index_type,
        )

        self.document_loader = DocumentLoader()
        self._is_indexed = False

    def index_documents(
        self,
        documents: List[Document],
        show_progress: bool = True,
    ):
        """
        Index a list of documents.

        Args:
            documents: List of Document objects.
            show_progress: Show progress bar.
        """
        if not documents:
            logger.warning("No documents to index")
            return

        # Extract texts
        texts = [doc.text for doc in documents]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} documents...")
        embeddings = self.embedder.embed_documents(texts, show_progress=show_progress)

        # Add to vector store
        self.vector_store.add_documents(
            texts=texts,
            embeddings=embeddings,
            metadata=[doc.metadata for doc in documents],
            ids=[doc.doc_id for doc in documents],
        )

        self._is_indexed = True
        logger.info(f"Indexed {len(documents)} documents")

    def index_training_data(
        self,
        train_dir: str,
        categories: Optional[List[str]] = None,
        include_prompts_examples: bool = True,
    ):
        """
        Index all training data from CSV files.

        Args:
            train_dir: Directory containing category CSV files.
            categories: Categories to index. None for all.
            include_prompts_examples: Also include examples from prompts.py.
        """
        # Load training data
        documents = self.document_loader.load_all_training_data(
            train_dir, categories
        )

        # Optionally add examples from prompts.py
        if include_prompts_examples:
            from ..classification.prompts import CATEGORIES

            for category in CATEGORIES.keys():
                if categories and category not in categories:
                    continue
                examples = self.document_loader.load_examples_from_prompts(category)
                documents.extend(examples)

        # Index
        self.index_documents(documents)

    def index_pdf(
        self,
        pdf_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Index a PDF document.

        Args:
            pdf_path: Path to PDF file.
            metadata: Optional additional metadata.
        """
        documents = self.document_loader.load_pdf(pdf_path, metadata)
        self.index_documents(documents)

    def index_text_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Index a text file.

        Args:
            file_path: Path to text file.
            metadata: Optional additional metadata.
        """
        documents = self.document_loader.load_text_file(file_path, metadata)
        self.index_documents(documents)

    def search(
        self,
        query: str,
        k: int = 5,
        category: Optional[str] = None,
        rating: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Query text.
            k: Number of results.
            category: Optional category filter.
            rating: Optional rating filter.

        Returns:
            List of result dicts with 'text', 'metadata', 'score'.
        """
        if not self._is_indexed:
            logger.warning("No documents indexed")
            return []

        # Embed query
        query_embedding = self.embedder.embed_query(query)

        # Build filter
        filter_fn = None
        if category or rating is not None:
            def filter_fn(meta):
                if category and meta.get("category") != category:
                    return False
                if rating is not None and meta.get("rating") != rating:
                    return False
                return True

        # Search
        results = self.vector_store.search(
            query_embedding,
            k=k,
            filter_fn=filter_fn,
        )

        return results

    def get_similar_examples(
        self,
        query: str,
        category: str,
        k: int = 5,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Get similar labeled examples for few-shot prompting.

        Args:
            query: Query text to find similar examples for.
            category: Category to search within.
            k: Number of examples to retrieve.
            min_score: Minimum similarity score threshold.

        Returns:
            List of example dicts with 'text', 'rating', 'score'.
        """
        results = self.search(query, k=k * 2, category=category)

        # Filter by score and format for few-shot
        examples = []
        for r in results:
            if r["score"] < min_score:
                continue

            rating = r["metadata"].get("rating")
            if rating is None:
                continue

            examples.append({
                "text": r["text"],
                "rating": rating,
                "score": r["score"],
            })

            if len(examples) >= k:
                break

        return examples

    def get_balanced_examples(
        self,
        query: str,
        category: str,
        n_per_class: int = 1,
        rating_classes: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get balanced examples across rating classes.

        Retrieves similar examples while ensuring representation
        from each rating class.

        Args:
            query: Query text.
            category: Category to search within.
            n_per_class: Number of examples per rating class.
            rating_classes: Rating classes to include. None for auto-detect.

        Returns:
            List of example dicts with 'text', 'rating', 'score'.
        """
        # Determine rating classes
        if rating_classes is None:
            # Get from category
            from ..classification.prompts import CATEGORIES
            cat_info = CATEGORIES.get(category, {})
            rating_scale = cat_info.get("rating_scale", {})
            rating_classes = sorted(rating_scale.keys())

        examples = []
        for rating in rating_classes:
            # Search for each rating class
            results = self.search(query, k=n_per_class * 2, category=category, rating=rating)

            for r in results[:n_per_class]:
                examples.append({
                    "text": r["text"],
                    "rating": rating,
                    "score": r["score"],
                })

        # Sort by rating for consistency
        examples.sort(key=lambda x: (x["rating"], -x["score"]))

        return examples

    def get_context_documents(
        self,
        query: str,
        k: int = 3,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get context documents (not examples) for RAG.

        Useful for retrieving relevant documentation or guidelines.

        Args:
            query: Query text.
            k: Number of documents.
            source_filter: Filter by source (e.g., "pdf", "manual").

        Returns:
            List of document dicts.
        """
        filter_fn = None
        if source_filter:
            filter_fn = lambda m: source_filter in m.get("source", "")

        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.search(query_embedding, k=k, filter_fn=filter_fn)

    def save(self, path: str):
        """
        Save retriever to disk.

        Args:
            path: Directory path to save to.
        """
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save vector store
        self.vector_store.save(str(path / "vector_store"))

        # Save config
        config = {
            "embedding_model": self.embedding_model_name,
            "index_type": self.index_type,
            "embedding_dim": self.embedder.embedding_dim,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved retriever to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        device: Optional[str] = None,
        use_gpu: bool = False,
    ) -> "RAGRetriever":
        """
        Load retriever from disk.

        Args:
            path: Directory path to load from.
            device: Device for embeddings.
            use_gpu: Use GPU for FAISS.

        Returns:
            Loaded RAGRetriever instance.
        """
        import json

        path = Path(path)

        # Load config
        with open(path / "config.json", "r") as f:
            config = json.load(f)

        # Create instance
        retriever = cls(
            embedding_model=config["embedding_model"],
            device=device,
            index_type=config["index_type"],
        )

        # Load vector store
        retriever.vector_store = VectorStore.load(
            str(path / "vector_store"),
            use_gpu=use_gpu,
        )
        retriever._is_indexed = True

        logger.info(f"Loaded retriever from {path}")
        return retriever

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the retriever."""
        return {
            "is_indexed": self._is_indexed,
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self.embedder.embedding_dim,
            **self.vector_store.get_stats(),
        }
