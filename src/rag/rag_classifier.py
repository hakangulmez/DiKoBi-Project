"""
RAG-enhanced classifier for DiKoBi.
Combines retrieval-augmented generation with few-shot classification.
"""

import logging
from typing import Optional, Dict, Any, List, Union
import torch

from ..classification.classifier import TextClassifier
from ..classification.prompts import CATEGORIES, SYSTEM_PROMPT, get_valid_outputs
from .retriever import RAGRetriever

logger = logging.getLogger(__name__)


class RAGClassifier:
    """
    RAG-enhanced text classifier for DiKoBi categories.

    Combines the base TextClassifier with RAG retrieval to:
    1. Dynamically retrieve similar examples from your data
    2. Include relevant context from documentation
    3. Provide better few-shot examples based on similarity

    Example:
        # Initialize
        classifier = RAGClassifier(
            model_name="Qwen/Qwen2.5-3B-Instruct",
            retriever_path="data/rag/index"
        )

        # Classify with RAG
        result = classifier.classify(
            category="1_D_M",
            text="Der Schüler zeigt kein Interesse",
            n_examples=3,
            use_balanced=True
        )

        # Batch classification
        results = classifier.classify_batch(
            category="1_D_M",
            texts=["text1", "text2", ...],
            n_examples=3
        )
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        retriever_path: Optional[str] = None,
        retriever: Optional[RAGRetriever] = None,
        use_8bit: bool = None,
        max_new_tokens: int = 10,
        temperature: float = 0.0,
        batch_size: int = None,
    ):
        """
        Initialize RAG classifier.

        Args:
            model_name: HuggingFace model ID.
            retriever_path: Path to saved RAG index. If None, must provide retriever.
            retriever: Pre-initialized RAGRetriever. If None, loads from path.
            use_8bit: Use 8-bit quantization.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            batch_size: Batch size for processing.
        """
        # Initialize base classifier
        self.classifier = TextClassifier(
            model_name=model_name,
            use_8bit=use_8bit,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=batch_size,
        )

        # Initialize or load retriever
        if retriever is not None:
            self.retriever = retriever
        elif retriever_path is not None:
            logger.info(f"Loading RAG retriever from {retriever_path}")
            self.retriever = RAGRetriever.load(retriever_path)
        else:
            logger.warning("No retriever provided. RAG features will be disabled.")
            self.retriever = None

        self.model_name = model_name

    def _build_rag_prompt(
        self,
        category: str,
        text: str,
        examples: List[Dict[str, Any]],
        context_docs: Optional[List[Dict[str, Any]]] = None,
        prompt_format: str = "standard",
    ) -> str:
        """
        Build prompt with RAG-retrieved examples and context.

        Args:
            category: DiKoBi category.
            text: Text to classify.
            examples: Retrieved similar examples.
            context_docs: Optional additional context documents.
            prompt_format: Prompt format style.

        Returns:
            Formatted prompt string.
        """
        cat_info = CATEGORIES.get(category, {})
        cat_name = cat_info.get("name", category)
        definition = cat_info.get("definition", "[Definition not available]")
        rating_scale = cat_info.get("rating_scale", {})

        # Build base prompt
        prompt = f"""{SYSTEM_PROMPT}

Category: {cat_name}

{definition}

Rating Scale:
"""
        # Add rating scale
        for rating, description in sorted(rating_scale.items()):
            prompt += f"{rating}: {description}\n"

        # Add context documents if available
        if context_docs:
            prompt += "\nRelevant Context:\n"
            for doc in context_docs:
                prompt += f"- {doc['text'][:200]}...\n" if len(doc['text']) > 200 else f"- {doc['text']}\n"

        # Add retrieved examples
        if examples:
            prompt += "\nSimilar Examples (Retrieved):\n\n"
            for i, ex in enumerate(examples, 1):
                score_info = f" (similarity: {ex.get('score', 0):.2f})" if 'score' in ex else ""
                prompt += f"Example {i}{score_info}:\n"
                prompt += f'Text: "{ex["text"]}"\n'
                prompt += f'Rating: {ex["rating"]}\n\n'

        # Add text to classify
        valid_outputs = get_valid_outputs(category)
        valid_range_str = ", ".join(str(x) for x in sorted(valid_outputs))

        prompt += f"""
Now analyze this teacher response and provide ONLY the rating number:

Text to rate:
"{text}"

Valid ratings: {valid_range_str}
Rating:"""

        return prompt

    def classify(
        self,
        category: str,
        text: str,
        n_examples: int = 3,
        use_balanced: bool = True,
        include_context: bool = False,
        n_context: int = 2,
        prompt_format: str = "standard",
        return_raw: bool = False,
        return_examples: bool = False,
    ) -> Dict[str, Any]:
        """
        Classify text using RAG-enhanced few-shot prompting.

        Args:
            category: DiKoBi category (e.g., '1_D_M').
            text: Input text to classify.
            n_examples: Number of similar examples to retrieve.
            use_balanced: Use balanced retrieval across rating classes.
            include_context: Include context documents in prompt.
            n_context: Number of context documents.
            prompt_format: Prompt format style.
            return_raw: Include raw model output.
            return_examples: Include retrieved examples in result.

        Returns:
            Dictionary with score, category, and optionally examples.
        """
        examples = []
        context_docs = None

        # Retrieve examples using RAG
        if self.retriever is not None:
            if use_balanced:
                # Get balanced examples across rating classes
                rating_scale = CATEGORIES.get(category, {}).get("rating_scale", {})
                n_per_class = max(1, n_examples // len(rating_scale))
                examples = self.retriever.get_balanced_examples(
                    query=text,
                    category=category,
                    n_per_class=n_per_class,
                    rating_classes=list(rating_scale.keys()),
                )
            else:
                # Get most similar examples
                examples = self.retriever.get_similar_examples(
                    query=text,
                    category=category,
                    k=n_examples,
                )

            # Get context documents if requested
            if include_context:
                context_docs = self.retriever.get_context_documents(
                    query=text,
                    k=n_context,
                )

        # Build RAG prompt
        prompt = self._build_rag_prompt(
            category=category,
            text=text,
            examples=examples,
            context_docs=context_docs,
            prompt_format=prompt_format,
        )

        # Generate response
        valid_outputs = get_valid_outputs(category)

        with torch.no_grad():
            outputs = self.classifier.pipeline(
                prompt,
                max_new_tokens=self.classifier.max_new_tokens,
                return_full_text=False,
            )

        raw_output = outputs[0]["generated_text"]
        score = self.classifier._extract_score(raw_output, valid_outputs)

        # Retry if parsing failed
        if score is None:
            score = self.classifier._retry_if_unparsed(prompt, valid_outputs)

        # Build result
        result = {
            "score": score,
            "category": category,
            "valid_range": valid_outputs,
            "n_examples_used": len(examples),
        }

        if return_raw:
            result["raw_output"] = raw_output

        if return_examples:
            result["examples"] = examples

        return result

    def classify_batch(
        self,
        category: str,
        texts: List[str],
        n_examples: int = 3,
        use_balanced: bool = True,
        include_context: bool = False,
        n_context: int = 2,
        prompt_format: str = "standard",
        return_raw: bool = False,
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple texts using RAG-enhanced few-shot prompting.

        Args:
            category: DiKoBi category.
            texts: List of input texts.
            n_examples: Number of similar examples per text.
            use_balanced: Use balanced retrieval.
            include_context: Include context documents.
            n_context: Number of context documents.
            prompt_format: Prompt format style.
            return_raw: Include raw model outputs.
            show_progress: Show progress bar.

        Returns:
            List of result dictionaries.
        """
        results = []
        valid_outputs = get_valid_outputs(category)

        # Setup progress
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc=f"RAG Classifying {category}")
            except ImportError:
                iterator = texts
                logger.info(f"Processing {len(texts)} texts...")
        else:
            iterator = texts

        # Build prompts with RAG for each text
        prompts = []
        for text in texts:
            examples = []
            context_docs = None

            if self.retriever is not None:
                if use_balanced:
                    rating_scale = CATEGORIES.get(category, {}).get("rating_scale", {})
                    n_per_class = max(1, n_examples // len(rating_scale))
                    examples = self.retriever.get_balanced_examples(
                        query=text,
                        category=category,
                        n_per_class=n_per_class,
                        rating_classes=list(rating_scale.keys()),
                    )
                else:
                    examples = self.retriever.get_similar_examples(
                        query=text,
                        category=category,
                        k=n_examples,
                    )

                if include_context:
                    context_docs = self.retriever.get_context_documents(
                        query=text,
                        k=n_context,
                    )

            prompt = self._build_rag_prompt(
                category=category,
                text=text,
                examples=examples,
                context_docs=context_docs,
                prompt_format=prompt_format,
            )
            prompts.append(prompt)

        # Process in batches
        with torch.no_grad():
            outputs = self.classifier.pipeline(
                prompts,
                max_new_tokens=self.classifier.max_new_tokens,
                return_full_text=False,
                batch_size=self.classifier.batch_size,
            )

        # Process outputs
        for i, output in enumerate(outputs):
            if isinstance(output, list):
                output = output[0]

            raw_output = output["generated_text"]
            score = self.classifier._extract_score(raw_output, valid_outputs)

            if score is None:
                score = self.classifier._retry_if_unparsed(prompts[i], valid_outputs)

            result = {
                "score": score,
                "category": category,
                "valid_range": valid_outputs,
                "parsing_failed": score is None,
            }

            if return_raw or score is None:
                result["raw_output"] = raw_output

            results.append(result)

        return results

    def classify_with_static_examples(
        self,
        category: str,
        text: str,
        examples: List[Dict[str, Any]],
        return_raw: bool = False,
    ) -> Dict[str, Any]:
        """
        Classify using provided static examples (no RAG retrieval).

        Useful when you want to use specific examples rather than retrieval.

        Args:
            category: DiKoBi category.
            text: Text to classify.
            examples: List of example dicts with 'text' and 'rating'.
            return_raw: Include raw model output.

        Returns:
            Result dictionary.
        """
        prompt = self._build_rag_prompt(
            category=category,
            text=text,
            examples=examples,
        )

        valid_outputs = get_valid_outputs(category)

        with torch.no_grad():
            outputs = self.classifier.pipeline(
                prompt,
                max_new_tokens=self.classifier.max_new_tokens,
                return_full_text=False,
            )

        raw_output = outputs[0]["generated_text"]
        score = self.classifier._extract_score(raw_output, valid_outputs)

        if score is None:
            score = self.classifier._retry_if_unparsed(prompt, valid_outputs)

        result = {
            "score": score,
            "category": category,
            "valid_range": valid_outputs,
        }

        if return_raw:
            result["raw_output"] = raw_output

        return result

    def get_retriever_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG retriever."""
        if self.retriever is None:
            return {"status": "no_retriever"}
        return self.retriever.get_stats()

    def clear_cache(self):
        """Clear GPU cache."""
        self.classifier.clear_cache()

    def unload(self):
        """Unload model and free resources."""
        self.classifier.unload()


def create_rag_classifier(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    train_dir: str = "data/processed/train",
    index_path: str = "data/rag/index",
    embedding_model: str = "multilingual-small",
    categories: Optional[List[str]] = None,
    force_reindex: bool = False,
) -> RAGClassifier:
    """
    Convenience function to create a RAG classifier with indexing.

    Creates the RAG index if it doesn't exist or if force_reindex is True.

    Args:
        model_name: LLM model name.
        train_dir: Training data directory.
        index_path: Path to save/load RAG index.
        embedding_model: Embedding model for RAG.
        categories: Categories to index. None for all.
        force_reindex: Force rebuilding the index.

    Returns:
        Initialized RAGClassifier.
    """
    from pathlib import Path

    index_dir = Path(index_path)

    # Check if index exists
    if index_dir.exists() and not force_reindex:
        logger.info(f"Loading existing RAG index from {index_path}")
        return RAGClassifier(
            model_name=model_name,
            retriever_path=index_path,
        )

    # Create new index
    logger.info(f"Creating new RAG index at {index_path}")

    # Initialize retriever
    retriever = RAGRetriever(embedding_model=embedding_model)

    # Index training data
    retriever.index_training_data(train_dir, categories)

    # Save index
    retriever.save(index_path)

    # Create classifier
    return RAGClassifier(
        model_name=model_name,
        retriever=retriever,
    )
