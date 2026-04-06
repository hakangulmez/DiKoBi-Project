"""
RAG experiment runner for DiKoBi.
Compares RAG-based classification with static few-shot approaches.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGExperimentRunner:
    """
    Run experiments comparing RAG vs static few-shot classification.

    Experiments:
    1. RAG vs Static: Compare RAG retrieval with static examples
    2. Retrieval Strategies: Compare balanced vs similarity-based retrieval
    3. N-Examples: Compare different numbers of retrieved examples
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        train_dir: str = "data/processed/train",
        test_dir: str = "data/processed/test",
        rag_index_path: str = "data/rag/index",
        results_dir: str = "results/rag",
    ):
        """
        Initialize experiment runner.

        Args:
            model_name: LLM model to use.
            train_dir: Training data directory.
            test_dir: Test data directory.
            rag_index_path: Path to RAG index.
            results_dir: Directory to save results.
        """
        self.model_name = model_name
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.rag_index_path = Path(rag_index_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Lazy loading
        self.rag_classifier = None
        self.static_classifier = None
        self.retriever = None

    def _ensure_rag_index(self, categories: List[str] = None):
        """Ensure RAG index exists."""
        from src.rag import RAGRetriever

        if not self.rag_index_path.exists():
            logger.info("Creating RAG index...")
            self.retriever = RAGRetriever()
            self.retriever.index_training_data(
                str(self.train_dir),
                categories=categories,
            )
            self.retriever.save(str(self.rag_index_path))
        else:
            logger.info(f"Loading RAG index from {self.rag_index_path}")
            self.retriever = RAGRetriever.load(str(self.rag_index_path))

    def _load_classifiers(self):
        """Load both RAG and static classifiers."""
        from src.rag import RAGClassifier
        from src.classification import TextClassifier

        if self.rag_classifier is None:
            logger.info("Loading RAG classifier...")
            self.rag_classifier = RAGClassifier(
                model_name=self.model_name,
                retriever=self.retriever,
            )

        if self.static_classifier is None:
            logger.info("Loading static classifier...")
            self.static_classifier = TextClassifier(
                model_name=self.model_name,
            )

    def _load_test_data(
        self,
        category: str,
        test_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load test data for a category."""
        test_file = self.test_dir / f"{category}.csv"
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")

        df = pd.read_csv(test_file)

        # Filter valid rows
        df = df[df[category].notna()].copy()

        if test_size:
            df = df.head(test_size)

        return df

    def run_rag_vs_static_experiment(
        self,
        category: str,
        test_size: int = 50,
        n_examples: int = 3,
    ) -> Dict[str, Any]:
        """
        Compare RAG vs static few-shot classification.

        Args:
            category: Category to test.
            test_size: Number of test samples.
            n_examples: Number of examples in prompts.

        Returns:
            Experiment results dictionary.
        """
        from RAGtests.mse_evaluator import MSEEvaluator

        self._ensure_rag_index([category])
        self._load_classifiers()

        # Load test data
        test_df = self._load_test_data(category, test_size)
        texts = test_df["free_text_answer"].tolist()
        true_ratings = test_df[category].astype(int).tolist()

        results = {
            "category": category,
            "test_size": len(texts),
            "n_examples": n_examples,
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "methods": {},
        }

        # 1. RAG Classification (balanced)
        logger.info("Running RAG (balanced) classification...")
        start_time = time.time()
        rag_results = self.rag_classifier.classify_batch(
            category=category,
            texts=texts,
            n_examples=n_examples,
            use_balanced=True,
        )
        rag_time = time.time() - start_time
        rag_predictions = [r["score"] for r in rag_results]

        evaluator = MSEEvaluator()
        rag_metrics = evaluator.calculate_all_metrics(true_ratings, rag_predictions)
        results["methods"]["rag_balanced"] = {
            "metrics": rag_metrics,
            "time_seconds": rag_time,
        }

        # 2. RAG Classification (similarity only)
        logger.info("Running RAG (similarity) classification...")
        start_time = time.time()
        rag_sim_results = self.rag_classifier.classify_batch(
            category=category,
            texts=texts,
            n_examples=n_examples,
            use_balanced=False,
        )
        rag_sim_time = time.time() - start_time
        rag_sim_predictions = [r["score"] for r in rag_sim_results]

        rag_sim_metrics = evaluator.calculate_all_metrics(true_ratings, rag_sim_predictions)
        results["methods"]["rag_similarity"] = {
            "metrics": rag_sim_metrics,
            "time_seconds": rag_sim_time,
        }

        # 3. Static Few-Shot (using predefined examples)
        logger.info("Running static few-shot classification...")
        start_time = time.time()
        static_results = self.static_classifier.classify_batch(
            category=category,
            texts=texts,
            prompt_template=f"few_shot_{min(n_examples, 3)}",
        )
        static_time = time.time() - start_time
        static_predictions = [r["score"] for r in static_results]

        static_metrics = evaluator.calculate_all_metrics(true_ratings, static_predictions)
        results["methods"]["static_few_shot"] = {
            "metrics": static_metrics,
            "time_seconds": static_time,
        }

        # 4. Zero-shot baseline
        logger.info("Running zero-shot classification...")
        start_time = time.time()
        zero_results = self.static_classifier.classify_batch(
            category=category,
            texts=texts,
            prompt_template="zero_shot",
        )
        zero_time = time.time() - start_time
        zero_predictions = [r["score"] for r in zero_results]

        zero_metrics = evaluator.calculate_all_metrics(true_ratings, zero_predictions)
        results["methods"]["zero_shot"] = {
            "metrics": zero_metrics,
            "time_seconds": zero_time,
        }

        # Log comparison
        logger.info("=" * 60)
        logger.info(f"Results for {category}:")
        logger.info("-" * 60)
        for method, data in results["methods"].items():
            mse = data["metrics"].get("mse", "N/A")
            acc = data["metrics"].get("accuracy", "N/A")
            logger.info(f"  {method:20s}: MSE={mse:.4f}, Accuracy={acc:.4f}")
        logger.info("=" * 60)

        return results

    def run_n_examples_experiment(
        self,
        category: str,
        test_size: int = 50,
        n_examples_list: List[int] = [1, 2, 3, 5, 7],
    ) -> Dict[str, Any]:
        """
        Compare different numbers of RAG examples.

        Args:
            category: Category to test.
            test_size: Number of test samples.
            n_examples_list: List of n_examples values to test.

        Returns:
            Experiment results dictionary.
        """
        from RAGtests.mse_evaluator import MSEEvaluator

        self._ensure_rag_index([category])
        self._load_classifiers()

        # Load test data
        test_df = self._load_test_data(category, test_size)
        texts = test_df["free_text_answer"].tolist()
        true_ratings = test_df[category].astype(int).tolist()

        results = {
            "category": category,
            "test_size": len(texts),
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "experiments": [],
        }

        evaluator = MSEEvaluator()

        for n_examples in n_examples_list:
            logger.info(f"Testing n_examples={n_examples}")

            start_time = time.time()
            rag_results = self.rag_classifier.classify_batch(
                category=category,
                texts=texts,
                n_examples=n_examples,
                use_balanced=True,
            )
            elapsed = time.time() - start_time

            predictions = [r["score"] for r in rag_results]
            metrics = evaluator.calculate_all_metrics(true_ratings, predictions)

            results["experiments"].append({
                "n_examples": n_examples,
                "metrics": metrics,
                "time_seconds": elapsed,
            })

            logger.info(f"  MSE: {metrics['mse']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        return results

    def run_full_comparison(
        self,
        categories: List[str] = None,
        test_size: int = 50,
    ) -> Dict[str, Any]:
        """
        Run full comparison across multiple categories.

        Args:
            categories: Categories to test. None for all.
            test_size: Number of test samples per category.

        Returns:
            Combined results dictionary.
        """
        # Determine categories
        if categories is None:
            categories = [f.stem for f in self.test_dir.glob("*.csv")]

        results = {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "test_size_per_category": test_size,
            "categories": {},
        }

        for category in categories:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing category: {category}")
            logger.info("=" * 60)

            try:
                cat_results = self.run_rag_vs_static_experiment(
                    category=category,
                    test_size=test_size,
                )
                results["categories"][category] = cat_results
            except Exception as e:
                logger.error(f"Failed for {category}: {e}")
                results["categories"][category] = {"error": str(e)}

        # Calculate summary
        summary = self._calculate_summary(results)
        results["summary"] = summary

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"rag_comparison_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nResults saved to {output_file}")

        return results

    def _calculate_summary(self, results: Dict) -> Dict:
        """Calculate summary statistics across categories."""
        methods = ["rag_balanced", "rag_similarity", "static_few_shot", "zero_shot"]
        summary = {method: {"mse": [], "accuracy": []} for method in methods}

        for category, cat_results in results.get("categories", {}).items():
            if "error" in cat_results:
                continue

            for method in methods:
                if method in cat_results.get("methods", {}):
                    metrics = cat_results["methods"][method]["metrics"]
                    if metrics.get("mse") is not None:
                        summary[method]["mse"].append(metrics["mse"])
                    if metrics.get("accuracy") is not None:
                        summary[method]["accuracy"].append(metrics["accuracy"])

        # Calculate averages
        for method in methods:
            mse_values = summary[method]["mse"]
            acc_values = summary[method]["accuracy"]
            summary[method] = {
                "avg_mse": np.mean(mse_values) if mse_values else None,
                "std_mse": np.std(mse_values) if mse_values else None,
                "avg_accuracy": np.mean(acc_values) if acc_values else None,
                "std_accuracy": np.std(acc_values) if acc_values else None,
                "n_categories": len(mse_values),
            }

        return summary

    def cleanup(self):
        """Cleanup resources."""
        if self.rag_classifier:
            self.rag_classifier.unload()
        if self.static_classifier:
            self.static_classifier.unload()


def main():
    """Run RAG experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG experiments")
    parser.add_argument(
        "--category",
        type=str,
        default="1_D_M",
        help="Category to test",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=50,
        help="Number of test samples",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="LLM model to use",
    )
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Run on all categories",
    )

    args = parser.parse_args()

    runner = RAGExperimentRunner(model_name=args.model)

    try:
        if args.all_categories:
            results = runner.run_full_comparison(test_size=args.test_size)
        else:
            results = runner.run_rag_vs_static_experiment(
                category=args.category,
                test_size=args.test_size,
            )

        # Print summary
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)

        if "summary" in results:
            print("\nSummary (averages across categories):")
            for method, stats in results["summary"].items():
                print(f"  {method}:")
                print(f"    MSE: {stats['avg_mse']:.4f} (+/- {stats['std_mse']:.4f})")
                print(f"    Accuracy: {stats['avg_accuracy']:.4f}")

    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
