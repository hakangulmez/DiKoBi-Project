"""
RAG (Retrieval-Augmented Generation) system for DiKoBi.

This module provides RAG capabilities to enhance the few-shot classification
by dynamically retrieving relevant examples and context from your own data.

Components:
- embeddings: Document embedding using sentence transformers
- vector_store: FAISS-based vector storage and retrieval
- retriever: High-level retriever for semantic search
- document_loader: Load documents from various formats
- rag_classifier: RAG-enhanced classifier combining retrieval with few-shot
"""

from .embeddings import EmbeddingModel
from .vector_store import VectorStore
from .retriever import RAGRetriever
from .document_loader import DocumentLoader
from .rag_classifier import RAGClassifier

__all__ = [
    "EmbeddingModel",
    "VectorStore",
    "RAGRetriever",
    "DocumentLoader",
    "RAGClassifier",
]
