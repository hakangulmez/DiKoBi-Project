"""
Embedding model for RAG system.
Uses sentence-transformers for high-quality text embeddings.
"""

import logging
from typing import List, Union, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Text embedding model using sentence-transformers.

    Supports multiple embedding models optimized for different use cases:
    - Multilingual models for German/English text
    - Different sizes for speed/quality tradeoffs

    Example:
        embedder = EmbeddingModel()
        embeddings = embedder.embed(["text1", "text2"])
    """

    # Recommended models for German educational text
    RECOMMENDED_MODELS = {
        "multilingual-small": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "multilingual-large": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "german-semantic": "deutsche-telekom/gbert-large-paraphrase-cosine",
        "e5-multilingual": "intfloat/multilingual-e5-base",
        "bge-m3": "BAAI/bge-m3",  # State-of-the-art multilingual
    }

    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize the embedding model.

        Args:
            model_name: HuggingFace model ID or key from RECOMMENDED_MODELS.
                       If None, uses DEFAULT_MODEL.
            device: Device to use ('cuda', 'mps', 'cpu'). None for auto-detect.
            normalize_embeddings: Whether to L2-normalize embeddings.
            batch_size: Batch size for encoding.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for RAG. "
                "Install with: pip install sentence-transformers"
            )

        # Resolve model name
        if model_name is None:
            model_name = self.DEFAULT_MODEL
        elif model_name in self.RECOMMENDED_MODELS:
            model_name = self.RECOMMENDED_MODELS[model_name]

        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

        # Auto-detect device
        if device is None:
            from ..utils.device import get_device
            device_obj = get_device()
            device = str(device_obj)  # SentenceTransformer accepts string or device
        self.device = device

        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Using device: {device}")

        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Embedding dimension: {self.embedding_dim}")

    def embed(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed text(s) into dense vectors.

        Args:
            texts: Single text or list of texts to embed.
            show_progress: Show progress bar for large batches.

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query text.

        Some models (like E5) require special prefixes for queries.
        This method handles that automatically.

        Args:
            query: Query text to embed.

        Returns:
            numpy array of shape (embedding_dim,)
        """
        # Handle E5 model query prefix
        if "e5" in self.model_name.lower():
            query = f"query: {query}"

        return self.embed(query)[0]

    def embed_documents(
        self,
        documents: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed documents for indexing.

        Some models (like E5) require special prefixes for documents.
        This method handles that automatically.

        Args:
            documents: List of document texts.
            show_progress: Show progress bar.

        Returns:
            numpy array of shape (n_documents, embedding_dim)
        """
        # Handle E5 model document prefix
        if "e5" in self.model_name.lower():
            documents = [f"passage: {doc}" for doc in documents]

        return self.embed(documents, show_progress=show_progress)

    def similarity(
        self,
        query: Union[str, np.ndarray],
        documents: Union[List[str], np.ndarray],
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.

        Args:
            query: Query text or embedding.
            documents: List of document texts or embeddings.

        Returns:
            numpy array of similarity scores.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # Embed if needed
        if isinstance(query, str):
            query_emb = self.embed_query(query).reshape(1, -1)
        else:
            query_emb = query.reshape(1, -1)

        if isinstance(documents, list) and isinstance(documents[0], str):
            doc_embs = self.embed_documents(documents, show_progress=False)
        else:
            doc_embs = documents

        similarities = cosine_similarity(query_emb, doc_embs)[0]
        return similarities

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
        }
