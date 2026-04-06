"""
Vector store for RAG system using FAISS.
Provides efficient similarity search over embedded documents.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store for efficient similarity search.

    Stores document embeddings and metadata, enabling fast retrieval
    of relevant documents based on semantic similarity.

    Example:
        store = VectorStore(embedding_dim=384)
        store.add_documents(
            texts=["doc1", "doc2"],
            embeddings=embeddings,
            metadata=[{"category": "1_D_M"}, {"category": "1_D_M"}]
        )
        results = store.search(query_embedding, k=5)
        store.save("./vector_store")
    """

    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "flat",
        use_gpu: bool = False,
    ):
        """
        Initialize vector store.

        Args:
            embedding_dim: Dimension of embeddings.
            index_type: FAISS index type ('flat', 'ivf', 'hnsw').
            use_gpu: Whether to use GPU for FAISS (requires faiss-gpu).
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss is required for RAG. "
                "Install with: pip install faiss-cpu (or faiss-gpu for GPU support)"
            )

        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.use_gpu = use_gpu

        # Initialize index
        self.index = self._create_index(embedding_dim, index_type)

        # Store metadata and texts
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.ids: List[str] = []

        # Track if index needs training (for IVF)
        self._is_trained = index_type == "flat"

        logger.info(f"Created {index_type} vector store with dim={embedding_dim}")

    def _create_index(self, dim: int, index_type: str):
        """Create FAISS index of specified type."""
        import faiss

        if index_type == "flat":
            # Exact search (best for small datasets)
            index = faiss.IndexFlatIP(dim)  # Inner product (cosine for normalized)
        elif index_type == "ivf":
            # Inverted file index (faster for large datasets)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, 100)  # 100 clusters
        elif index_type == "hnsw":
            # Hierarchical NSW (good speed/accuracy tradeoff)
            index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors per node
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Move to GPU if requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("Using GPU for FAISS")
            except Exception as e:
                logger.warning(f"Could not use GPU for FAISS: {e}")

        return index

    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ):
        """
        Add documents to the vector store.

        Args:
            texts: List of document texts.
            embeddings: numpy array of shape (n_docs, embedding_dim).
            metadata: Optional list of metadata dicts for each document.
            ids: Optional list of unique IDs. Auto-generated if not provided.
        """
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have same length")

        # Generate IDs if not provided
        if ids is None:
            start_id = len(self.documents)
            ids = [f"doc_{start_id + i}" for i in range(len(texts))]

        # Generate empty metadata if not provided
        if metadata is None:
            metadata = [{} for _ in texts]

        # Ensure embeddings are float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

        # Train index if needed (for IVF)
        if not self._is_trained and self.index_type == "ivf":
            if len(embeddings) >= 100:  # Need enough vectors to train
                self.index.train(embeddings)
                self._is_trained = True
            else:
                logger.warning("Not enough vectors to train IVF index, using flat search")
                import faiss
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self._is_trained = True

        # Add to index
        self.index.add(embeddings)

        # Store documents and metadata
        self.documents.extend(texts)
        self.metadata.extend(metadata)
        self.ids.extend(ids)

        logger.info(f"Added {len(texts)} documents. Total: {len(self.documents)}")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_fn: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector.
            k: Number of results to return.
            filter_fn: Optional function to filter results by metadata.
                      Takes metadata dict, returns bool.

        Returns:
            List of result dicts with 'text', 'metadata', 'score', 'id'.
        """
        if len(self.documents) == 0:
            return []

        # Ensure query is 2D float32
        query = np.ascontiguousarray(
            query_embedding.reshape(1, -1).astype(np.float32)
        )

        # Search more if filtering (need larger pool when filtering by category/rating)
        # With 38k+ docs across 26 categories, need to search ~5% of corpus to find matches
        search_k = min(k * 200, len(self.documents)) if filter_fn else k

        # Search
        scores, indices = self.index.search(query, min(search_k, len(self.documents)))
        scores = scores[0]
        indices = indices[0]

        # Build results
        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue

            result = {
                "id": self.ids[idx],
                "text": self.documents[idx],
                "metadata": self.metadata[idx],
                "score": float(score),
            }

            # Apply filter
            if filter_fn and not filter_fn(result["metadata"]):
                continue

            results.append(result)

            if len(results) >= k:
                break

        return results

    def search_by_category(
        self,
        query_embedding: np.ndarray,
        category: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents within a specific category.

        Args:
            query_embedding: Query embedding vector.
            category: Category to filter by (e.g., "1_D_M").
            k: Number of results to return.

        Returns:
            List of result dicts.
        """
        return self.search(
            query_embedding,
            k=k,
            filter_fn=lambda m: m.get("category") == category,
        )

    def search_by_rating(
        self,
        query_embedding: np.ndarray,
        rating: int,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents with a specific rating.

        Args:
            query_embedding: Query embedding vector.
            rating: Rating to filter by (e.g., 0, 1, 2).
            k: Number of results to return.

        Returns:
            List of result dicts.
        """
        return self.search(
            query_embedding,
            k=k,
            filter_fn=lambda m: m.get("rating") == rating,
        )

    def get_all_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all documents for a specific category."""
        results = []
        for i, (doc, meta, doc_id) in enumerate(
            zip(self.documents, self.metadata, self.ids)
        ):
            if meta.get("category") == category:
                results.append({
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta,
                })
        return results

    def save(self, path: str):
        """
        Save vector store to disk.

        Args:
            path: Directory path to save to.
        """
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = path / "index.faiss"
        if self.use_gpu:
            # Convert back to CPU for saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_path))
        else:
            faiss.write_index(self.index, str(index_path))

        # Save documents and metadata
        data = {
            "documents": self.documents,
            "metadata": self.metadata,
            "ids": self.ids,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
        }
        data_path = path / "data.json"
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved vector store to {path}")

    @classmethod
    def load(cls, path: str, use_gpu: bool = False) -> "VectorStore":
        """
        Load vector store from disk.

        Args:
            path: Directory path to load from.
            use_gpu: Whether to use GPU for FAISS.

        Returns:
            Loaded VectorStore instance.
        """
        import faiss

        path = Path(path)

        # Load data
        data_path = path / "data.json"
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Create instance
        store = cls(
            embedding_dim=data["embedding_dim"],
            index_type=data["index_type"],
            use_gpu=use_gpu,
        )

        # Load FAISS index
        index_path = path / "index.faiss"
        store.index = faiss.read_index(str(index_path))

        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                store.index = faiss.index_cpu_to_gpu(res, 0, store.index)
            except Exception as e:
                logger.warning(f"Could not use GPU for FAISS: {e}")

        # Restore documents and metadata
        store.documents = data["documents"]
        store.metadata = data["metadata"]
        store.ids = data["ids"]
        store._is_trained = True

        logger.info(f"Loaded vector store from {path} ({len(store.documents)} documents)")

        return store

    def __len__(self) -> int:
        """Return number of documents in store."""
        return len(self.documents)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        categories = {}
        ratings = {}

        for meta in self.metadata:
            cat = meta.get("category", "unknown")
            rating = meta.get("rating", "unknown")

            categories[cat] = categories.get(cat, 0) + 1
            ratings[rating] = ratings.get(rating, 0) + 1

        return {
            "total_documents": len(self.documents),
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "categories": categories,
            "ratings": ratings,
        }
