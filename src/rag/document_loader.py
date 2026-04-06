"""
Document loader for RAG system.
Loads documents from various formats for indexing.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
import pandas as pd

logger = logging.getLogger(__name__)


class Document:
    """Simple document container."""

    def __init__(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ):
        self.text = text
        self.metadata = metadata or {}
        self.doc_id = doc_id

    def __repr__(self):
        return f"Document(text='{self.text[:50]}...', metadata={self.metadata})"


class DocumentLoader:
    """
    Load documents from various sources for RAG indexing.

    Supports:
    - CSV files (training data)
    - Text files
    - PDF files (optional, requires pypdf)
    - Custom document lists

    Example:
        loader = DocumentLoader()

        # Load from training CSV
        docs = loader.load_training_data(
            "data/processed/train/1_D_M.csv",
            category="1_D_M"
        )

        # Load from PDF manual
        docs = loader.load_pdf("docs/coding_manual.pdf")

        # Load custom documents
        docs = loader.load_documents([
            {"text": "example text", "rating": 1, "category": "1_D_M"}
        ])
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document loader.

        Args:
            chunk_size: Maximum characters per chunk for long documents.
            chunk_overlap: Overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_training_data(
        self,
        csv_path: str,
        category: str,
        text_column: str = "free_text_answer",
        rating_column: Optional[str] = None,
    ) -> List[Document]:
        """
        Load training data from CSV file.

        Args:
            csv_path: Path to CSV file.
            category: Category name (added to metadata).
            text_column: Column containing text.
            rating_column: Column containing ratings. If None, uses category name.

        Returns:
            List of Document objects.
        """
        df = pd.read_csv(csv_path)

        if rating_column is None:
            rating_column = category

        documents = []
        for idx, row in df.iterrows():
            text = str(row[text_column]).strip()
            if not text or text == "nan":
                continue

            # Get rating if available
            rating = None
            if rating_column in df.columns:
                rating_val = row[rating_column]
                if pd.notna(rating_val):
                    rating = int(rating_val)

            metadata = {
                "category": category,
                "source": str(csv_path),
                "row_index": idx,
            }
            if rating is not None:
                metadata["rating"] = rating

            doc = Document(
                text=text,
                metadata=metadata,
                doc_id=f"{category}_{idx}",
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from {csv_path}")
        return documents

    def load_all_training_data(
        self,
        train_dir: str,
        categories: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Load training data from all category CSV files.

        Args:
            train_dir: Directory containing category CSV files.
            categories: List of categories to load. If None, loads all.

        Returns:
            List of Document objects from all categories.
        """
        train_path = Path(train_dir)
        all_documents = []

        # Find all CSV files
        csv_files = list(train_path.glob("*.csv"))

        for csv_file in csv_files:
            # Extract category from filename
            category = csv_file.stem

            # Skip if not in requested categories
            if categories and category not in categories:
                continue

            try:
                docs = self.load_training_data(
                    str(csv_file),
                    category=category,
                )
                all_documents.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")

        logger.info(f"Loaded {len(all_documents)} documents from {len(csv_files)} files")
        return all_documents

    def load_text_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True,
    ) -> List[Document]:
        """
        Load document from text file.

        Args:
            file_path: Path to text file.
            metadata: Optional metadata to add.
            chunk: Whether to chunk long documents.

        Returns:
            List of Document objects.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        base_metadata = {"source": str(file_path)}
        if metadata:
            base_metadata.update(metadata)

        if chunk and len(content) > self.chunk_size:
            return self._chunk_text(content, base_metadata, Path(file_path).stem)
        else:
            return [Document(text=content, metadata=base_metadata)]

    def load_pdf(
        self,
        pdf_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True,
    ) -> List[Document]:
        """
        Load document from PDF file.

        Args:
            pdf_path: Path to PDF file.
            metadata: Optional metadata to add.
            chunk: Whether to chunk long documents.

        Returns:
            List of Document objects.

        Requires:
            pip install pypdf
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required to load PDFs. Install with: pip install pypdf"
            )

        reader = PdfReader(pdf_path)
        documents = []

        base_metadata = {"source": str(pdf_path), "type": "pdf"}
        if metadata:
            base_metadata.update(metadata)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text.strip():
                continue

            page_metadata = {**base_metadata, "page": page_num + 1}

            if chunk and len(text) > self.chunk_size:
                page_docs = self._chunk_text(
                    text,
                    page_metadata,
                    f"{Path(pdf_path).stem}_p{page_num + 1}",
                )
                documents.extend(page_docs)
            else:
                doc = Document(
                    text=text,
                    metadata=page_metadata,
                    doc_id=f"{Path(pdf_path).stem}_p{page_num + 1}",
                )
                documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from {pdf_path}")
        return documents

    def load_documents(
        self,
        documents: List[Dict[str, Any]],
        text_key: str = "text",
    ) -> List[Document]:
        """
        Load documents from a list of dictionaries.

        Args:
            documents: List of dicts with at least a 'text' key.
            text_key: Key containing the text content.

        Returns:
            List of Document objects.
        """
        result = []
        for i, doc in enumerate(documents):
            text = doc.get(text_key, "")
            if not text:
                continue

            # All other keys become metadata
            metadata = {k: v for k, v in doc.items() if k != text_key}

            result.append(Document(
                text=text,
                metadata=metadata,
                doc_id=doc.get("id", f"doc_{i}"),
            ))

        return result

    def load_examples_from_prompts(self, category: str) -> List[Document]:
        """
        Load few-shot examples defined in prompts.py.

        Args:
            category: Category name.

        Returns:
            List of Document objects from predefined examples.
        """
        from ..classification.prompts import CATEGORIES

        if category not in CATEGORIES:
            logger.warning(f"Category {category} not found")
            return []

        cat_info = CATEGORIES[category]
        examples = cat_info.get("examples", [])

        documents = []
        for i, example in enumerate(examples):
            doc = Document(
                text=example["text"],
                metadata={
                    "category": category,
                    "rating": example["rating"],
                    "source": "prompts.py",
                    "is_example": True,
                },
                doc_id=f"{category}_example_{i}",
            )
            documents.append(doc)

        return documents

    def _chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any],
        base_id: str,
    ) -> List[Document]:
        """
        Split long text into overlapping chunks.

        Args:
            text: Text to chunk.
            metadata: Metadata to attach to all chunks.
            base_id: Base document ID.

        Returns:
            List of Document chunks.
        """
        chunks = []
        start = 0
        chunk_num = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end near chunk boundary
                for sep in [". ", ".\n", "! ", "? "]:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_metadata = {**metadata, "chunk": chunk_num}
                chunks.append(Document(
                    text=chunk_text,
                    metadata=chunk_metadata,
                    doc_id=f"{base_id}_chunk_{chunk_num}",
                ))
                chunk_num += 1

            start = end - self.chunk_overlap

        return chunks


def load_dikobi_training_data(
    train_dir: str = "data/processed/train",
    categories: Optional[List[str]] = None,
) -> List[Document]:
    """
    Convenience function to load DiKoBi training data.

    Args:
        train_dir: Path to training data directory.
        categories: Categories to load. None for all.

    Returns:
        List of Document objects.
    """
    loader = DocumentLoader()
    return loader.load_all_training_data(train_dir, categories)
