"""Experiment Tracking.

Tracks all experiments (template comparison, prompt development, model comparison)
in a single format.

Additionally writes per-run artifacts into a dedicated folder under `data/experiments/`.
Each run gets its own folder:
`data/experiments/experiment_<YYYYMMDD_HHMMSS>__<category>__<experiment_id>/`

Inside the folder, the tracker writes:
- `predictions.csv` (per-sample predictions)
- `experiment.json` (per-experiment snapshot)
- `bundle.json` (experiment snapshot + per-sample predictions)
"""

import json
import csv
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class ExperimentTracker:
    """Unified tracker for all development experiments."""
    
    def __init__(self, category: str):
        """
        Initialize tracker for a specific category.
        
        Args:
            category: Category name (e.g., '1_D_M')
        """
        self.category = category
        
        # Directory for per-run artifacts (predictions CSV, experiment JSON) - git-ignored
        self.experiments_dir = Path("data/experiments")
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Directory for category history files - git-tracked
        self.history_dir = Path("results/experiment_history")
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.history_dir / f"{category}.json"
        
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Load experiment history from file."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_history(self):
        """Save experiment history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def _create_run_dir(self, timestamp: datetime, experiment_id: str) -> Path:
        """Create a dedicated folder for a single experiment run."""
        ts = timestamp.strftime("%Y%m%d_%H%M%S")
        run_dir = self.experiments_dir / f"experiment_{ts}__{self.category}__{experiment_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        existing_ids = [e.get('experiment_id', '') for e in self.history]
        # Extract numbers from existing IDs
        numbers = []
        for eid in existing_ids:
            if eid.startswith('exp_'):
                try:
                    numbers.append(int(eid.split('_')[1]))
                except (IndexError, ValueError):
                    pass
        
        next_num = max(numbers, default=0) + 1
        return f"exp_{next_num:03d}"
    
    def _convert_to_python(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            if obj.size == 1:
                return obj.item()
            else:
                return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_python(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._convert_to_python(v) for k, v in obj.items()}
        else:
            return obj

    @staticmethod
    def _to_int_or_none(value):
        """Best-effort conversion to int for robust comparisons/logging."""
        if value is None:
            return None
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            try:
                if np.isnan(value):
                    return None
            except Exception:
                pass
            try:
                # Ratings are ordinal integers in this project.
                return int(value)
            except Exception:
                return None
        if isinstance(value, str):
            s = value.strip()
            if not s or s.lower() in {"nan", "none", "null"}:
                return None
            try:
                return int(float(s))
            except Exception:
                return None
        # Unknown type
        return None

    @classmethod
    def _is_correct(cls, y_true, y_pred) -> Optional[bool]:
        """Return True/False if both labels are present, else None."""
        true_int = cls._to_int_or_none(y_true)
        pred_int = cls._to_int_or_none(y_pred)
        if true_int is None or pred_int is None:
            return None
        return true_int == pred_int
    
    def check_duplicate(
        self,
        model: str,
        prompt_template: str,
        max_samples: Optional[int] = None,
        use_8bit: bool = False,
        verbose: bool = True,
        max_new_tokens: int = None,
        output_format: Optional[str] = None,
    ) -> Dict:
        """
        Check if an experiment with these parameters already exists.
        
        This method handles all preparation internally:
        1. Load test data to determine samples_used
        2. Resolve model name to HuggingFace ID
        3. Get prompt text from category + template
        4. Build generation params with sensible defaults
        5. Check history for matching experiment
        
        Args:
            model: Model key (e.g., "qwen2.5-7b-text") or HuggingFace ID
            prompt_template: Template identifier (e.g., "zero_shot", "few_shot")
            max_samples: Maximum samples to use (None = all)
            use_8bit: Whether 8-bit quantization is used
            verbose: Whether to print duplicate detection message
            max_new_tokens: Maximum tokens for generation (None = auto-detect from notebook config)
            output_format: Output format - "json" or "text" (None = auto-detect from model config)
        
        Returns:
            Dict with:
                - 'cached': Existing experiment dict if duplicate found, None otherwise
                - 'model_hf_id': Resolved HuggingFace ID
                - 'quantization': Quantization string ("8bit"/"none")
                - 'output_format': Output format string ("json"/"text")
                - 'prompt_text': Full prompt text for this category
                - 'dataset_path': Path to test dataset
                - 'generation_params': Generation parameters dict
        """
        import pandas as pd
        from ..models import get_model_config
        from ..classification.prompts import get_prompt
        
        # Determine dataset path
        dataset_path = Path("data/processed/test") / f"{self.category}.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Test data not found: {dataset_path}")
        
        # Load test data to determine sample count
        df = pd.read_csv(dataset_path)
        total_samples = len(df)
        samples_used = min(max_samples, total_samples) if max_samples else total_samples
        
        # Resolve model name to HF ID and get configuration
        try:
            model_config = get_model_config(model)
            model_hf_id = model_config.hf_id
            # Only use model config if output_format not explicitly provided
            if output_format is None:
                output_format = model_config.output_format
        except:
            # If not in registry, assume it's already an HF ID
            model_hf_id = model
            # Only default to text if not explicitly provided
            if output_format is None:
                output_format = "text"
        
        # Get max_new_tokens from notebook config if not provided
        if max_new_tokens is None:
            try:
                import __main__
                if output_format == "json":
                    max_new_tokens = getattr(__main__, 'JSON_MAX_TOKENS', None)
                    if max_new_tokens is None:
                        raise ValueError(
                            f"JSON_MAX_TOKENS not defined in notebook configuration.\n"
                            f"Model '{model}' uses JSON output format.\n"
                            f"Add 'JSON_MAX_TOKENS = <value>' to your config cell."
                        )
                else:
                    max_new_tokens = getattr(__main__, 'TEXT_MAX_TOKENS', None)
                    if max_new_tokens is None:
                        raise ValueError(
                            f"TEXT_MAX_TOKENS not defined in notebook configuration.\n"
                            f"Model '{model}' uses Text output format.\n"
                            f"Add 'TEXT_MAX_TOKENS = <value>' to your config cell."
                        )
            except AttributeError:
                raise ValueError(
                    f"Cannot access notebook configuration.\n"
                    f"Make sure you're running this in a Jupyter notebook with the config cell executed."
                )
        
        # Get prompt text for this category + template
        sample_text = "[SAMPLE TEXT]"
        
        # Handle RAG templates specially (they don't exist in prompts.py)
        if prompt_template in ["rag_balanced", "rag_similarity"]:
            # For RAG, reconstruct the descriptor based on n_examples parameter
            try:
                import __main__
                n_examples = getattr(__main__, 'RAG_N_EXAMPLES', 3)
            except (AttributeError, ImportError):
                n_examples = 3  # Default fallback
            
            if prompt_template == "rag_balanced":
                prompt_text = f"[RAG Balanced] Dynamic examples from retriever (n={n_examples}, balanced by rating)"
            else:  # rag_similarity
                prompt_text = f"[RAG Similarity] Dynamic examples from retriever (n={n_examples}, similarity-based)"
        else:
            # Regular templates: generate prompt normally
            prompt_text = get_prompt(self.category, sample_text, template_name=prompt_template, output_format=output_format)
        
        # Build generation params
        generation_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False
        }
        
        # Quantization string
        quantization = "8bit" if use_8bit else "none"
        
        # Reload history to pick up experiments from other processes
        self.history = self._load_history()
        
        # Check for duplicates based on all deterministic configuration
        cached_experiment = None
        for entry in self.history:
            # Get existing generation params
            existing_gen_params = entry.get('generation_params', {})
            
            # Skip experiments without generation_params (can't verify all parameters)
            if not existing_gen_params:
                continue
            
            # # Debug: Show comparison for each entry
            # if verbose and entry.get("model") == model_hf_id and entry.get("prompt_template") == prompt_template:
            #     print(f"\n[DEBUG] Checking experiment {entry.get('experiment_id')}:")
            #     print(f"  Quantization: {entry.get('quantization')} == {quantization} ? {entry.get('quantization') == quantization}")
            #     print(f"  Output format: {entry.get('output_format')} == {output_format} ? {entry.get('output_format') == output_format}")
            #     print(f"  Samples: {entry['dataset']['samples_used']} == {samples_used} ? {entry['dataset']['samples_used'] == samples_used}")
            #     print(f"  Dataset path: {entry['dataset'].get('path')} == {str(dataset_path)} ? {entry['dataset'].get('path') == str(dataset_path)}")
            #     print(f"  max_new_tokens: {existing_gen_params.get('max_new_tokens')} == {generation_params.get('max_new_tokens')} ? {existing_gen_params.get('max_new_tokens') == generation_params.get('max_new_tokens')}")
            #     print(f"  temperature: {existing_gen_params.get('temperature')} == {generation_params.get('temperature')} ? {existing_gen_params.get('temperature') == generation_params.get('temperature')}")
            #     print(f"  top_p: {existing_gen_params.get('top_p')} == {generation_params.get('top_p')} ? {existing_gen_params.get('top_p') == generation_params.get('top_p')}")
            #     print(f"  do_sample: {existing_gen_params.get('do_sample')} == {generation_params.get('do_sample')} ? {existing_gen_params.get('do_sample') == generation_params.get('do_sample')}")
            #     print(f"  Prompt text match: {entry.get('prompt_text') == prompt_text}")
            #     if entry.get('prompt_text') != prompt_text:
            #         import hashlib
            #         print(f"    Expected prompt hash: {hashlib.md5(prompt_text.encode('utf-8')).hexdigest()[:8]}")
            #         print(f"    Found prompt hash: {hashlib.md5(entry.get('prompt_text', '').encode('utf-8')).hexdigest()[:8]}")
            
            # Check if all key parameters match
            # Normalize paths for cross-platform compatibility (Windows uses \, Unix uses /)
            stored_path = entry["dataset"]["path"].replace("\\", "/")
            current_path = str(dataset_path).replace("\\", "/")
            
            if (
                entry.get("model") == model_hf_id
                and entry.get("prompt_template") == prompt_template
                and entry.get("prompt_text") == prompt_text
                and entry.get("quantization") == quantization
                and entry.get("output_format") == output_format
                and entry["dataset"].get("max_samples") == max_samples  # Compare input config, not output
                and stored_path == current_path
                and existing_gen_params.get("max_new_tokens") == generation_params.get("max_new_tokens")
                and existing_gen_params.get("temperature") == generation_params.get("temperature")
                and existing_gen_params.get("top_p") == generation_params.get("top_p")
                and existing_gen_params.get("do_sample") == generation_params.get("do_sample")
            ):
                # Found duplicate
                cached_experiment = entry
                if verbose:
                    metrics = entry.get('metrics', {})
                    date = entry.get('timestamp', 'N/A')[:19].replace('T', ' ')
                    model_hf_id_display = entry.get('model', 'N/A')  # Show full HF ID
                    
                    print(f"\n{'='*130}")
                    print(f"⚠️  DUPLICATE DETECTED - Using cached results")
                    print(f"{'='*130}")
                    print(f"\n{'Parameter':<20} {'Value':<60} {'Status':<50}")
                    print(f"{'-'*130}")
                    print(f"{'Experiment ID':<20} {entry.get('experiment_id', 'N/A'):<60} {'✓ Found in history':<50}")
                    print(f"{'Model (HF ID)':<20} {model_hf_id_display:<60} {'✓ Matches':<50}")
                    print(f"{'Template':<20} {entry.get('prompt_template', 'N/A'):<60} {'✓ Matches':<50}")
                    print(f"{'Quantization':<20} {entry.get('quantization', 'N/A'):<60} {'✓ Matches':<50}")
                    print(f"{'Output Format':<20} {entry.get('output_format', 'N/A'):<60} {'✓ Matches':<50}")
                    print(f"{'Max Tokens':<20} {existing_gen_params.get('max_new_tokens', 'N/A'):<60} {'✓ Matches':<50}")
                    print(f"{'Temperature':<20} {existing_gen_params.get('temperature', 'N/A'):<60} {'✓ Matches':<50}")
                    print(f"{'Top-p':<20} {existing_gen_params.get('top_p', 'N/A'):<60} {'✓ Matches':<50}")
                    print(f"{'Do Sample':<20} {existing_gen_params.get('do_sample', 'N/A'):<60} {'✓ Matches':<50}")
                    print(f"{'Samples Used':<20} {entry['dataset']['samples_used']:<60} {'✓ Matches':<50}")
                    print(f"{'Dataset Path':<20} {entry['dataset'].get('path', 'N/A').split('/')[-1]:<60} {'✓ Matches':<50}")
                    print(f"{'-'*130}")
                    print(f"\n{'Metric':<20} {'Value':<60} {'Recorded':<50}")
                    print(f"{'-'*130}")
                    print(f"{'QWK':<20} {metrics.get('qwk', 0):.4f}{'':<56} {date:<50}")
                    print(f"{'Accuracy':<20} {metrics.get('accuracy', 0):.2%}{'':<54} {'(deterministic with temp=0)':<50}")
                    print(f"{'MSE':<20} {metrics.get('mse', 0):.4f}{'':<56} {'':<50}")
                    print(f"{'MAE':<20} {metrics.get('mae', 0):.4f}{'':<56} {'':<50}")
                    print(f"{'='*130}")
                    print(f"\n💡 All parameters match → Results are identical (deterministic)")
                    print(f"   To run anyway, change any parameter (prompt, model, temperature, etc.)\n")
                break
        
        # Return computed metadata (always) plus cached experiment (if found)
        return {
            'cached': cached_experiment,
            'model_hf_id': model_hf_id,
            'quantization': quantization,
            'output_format': output_format,
            'prompt_text': prompt_text,
            'dataset_path': str(dataset_path),
            'generation_params': generation_params,
        }
    
    def log_experiment(
        self,
        model: str,
        prompt_template: str,
        prompt_text: str,
        metrics: Dict,
        dataset_path: str,
        total_samples: int,
        samples_used: int,
        generation_params: Dict,
        output_format: str,
        max_samples: Optional[int] = None,
        quantization: str = "none",
        version_name: Optional[str] = None,
        notes: str = "",
        predictions: Optional[List[Dict]] = None,
    ) -> str:
        """
        Log a new experiment to history.
        
        Args:
            model: Model name
            prompt_template: Template identifier
            prompt_text: Full prompt text
            metrics: Dictionary with accuracy, qwk, mse, mae
            dataset_path: Path to dataset file
            total_samples: Total samples in dataset
            samples_used: Number of samples actually used
            generation_params: Dict with max_new_tokens, temperature, top_p, do_sample (required)
            output_format: Output format - "json" or "text" (required)
            max_samples: Maximum samples configured (None = all)
            quantization: Quantization method used
            version_name: Optional custom version name
            notes: Optional notes about this experiment
        
        Returns:
            Experiment ID
        """
        
        # Check for duplicates using centralized method
        # Note: We use a slightly different approach here since we already have
        # all the prepared data (prompt_text, samples_used, etc.)
        # But we still check using the same logic
        
        # Reload history to pick up experiments from other processes
        self.history = self._load_history()
        
        # Check for duplicates based on all deterministic configuration
        for entry in self.history:
            # Get existing generation params (with fallback for old experiments)
            existing_gen_params = entry.get('generation_params', {})
            
            # Normalize paths for cross-platform compatibility
            stored_path = entry["dataset"]["path"].replace("\\", "/")
            current_path = str(dataset_path).replace("\\", "/")
            
            # Check if all key parameters match
            if (
                entry.get("model") == model
                and entry.get("prompt_template") == prompt_template
                and entry.get("prompt_text") == prompt_text
                and entry.get("quantization") == quantization
                and entry.get("output_format") == output_format
                and entry["dataset"].get("max_samples") == max_samples  # Compare input config, not output
                and stored_path == current_path
                and existing_gen_params.get("max_new_tokens") == generation_params.get("max_new_tokens")
                and existing_gen_params.get("temperature") == generation_params.get("temperature")
                and existing_gen_params.get("top_p") == generation_params.get("top_p")
                and existing_gen_params.get("do_sample") == generation_params.get("do_sample")
            ):
                # Found duplicate - return existing ID
                return entry.get('experiment_id')
        
        # No duplicate found - create new experiment entry
        experiment_id = self._generate_experiment_id()
        
        # Generate prompt hash for distinguishing different prompts with same template
        prompt_hash = hashlib.md5(prompt_text.encode('utf-8')).hexdigest()[:8]
        
        # Determine dataset split from path (normalize for Windows paths)
        split = "unknown"
        normalized_dataset_path = str(dataset_path).replace("\\", "/")
        if "/test/" in normalized_dataset_path:
            split = "test"
        elif "/train/" in normalized_dataset_path:
            split = "train"
        
        artifacts: Dict[str, str] = {}

        now = datetime.now()
        run_dir = self._create_run_dir(now, experiment_id)

        # Optional per-sample predictions CSV
        if predictions:
            pred_path = run_dir / "predictions.csv"
            fieldnames = [
                "sample_index",
                "text",
                "y_true",
                "y_pred",
                "parse_ok",
                "is_correct",
                "raw_output",
            ]
            with open(pred_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in predictions:
                    is_correct = self._is_correct(row.get("y_true"), row.get("y_pred"))
                    writer.writerow(
                        {
                            "sample_index": row.get("sample_index"),
                            "text": row.get("text", ""),
                            "y_true": row.get("y_true"),
                            "y_pred": row.get("y_pred"),
                            "parse_ok": row.get("parse_ok"),
                            "is_correct": is_correct,
                            "raw_output": row.get("raw_output", ""),
                        }
                    )
            artifacts["predictions_csv"] = str(pred_path)
        
        experiment = {
            "experiment_id": experiment_id,
            "version": version_name or experiment_id,
            "timestamp": now.isoformat(),
            "category": self.category,
            "model": model,
            "quantization": quantization,
            "output_format": output_format,
            "generation_params": generation_params,
            "prompt_hash": prompt_hash,  # For distinguishing different prompts
            "prompt_template": prompt_template,
            "prompt_text": prompt_text,
            "dataset": {
                "path": dataset_path,
                "split": split,
                "total_samples": total_samples,
                "samples_used": samples_used,
                "max_samples": max_samples
            },
            "metrics": self._convert_to_python(metrics),
            "notes": notes,
            "artifacts": artifacts,
        }

        # Per-experiment JSON snapshot (stable artifact, separate from category history)
        exp_path = run_dir / "experiment.json"
        with open(exp_path, "w", encoding="utf-8") as f:
            json.dump(experiment, f, indent=2, ensure_ascii=False)
        experiment["artifacts"]["experiment_json"] = str(exp_path)

        # Per-experiment bundle JSON (single file containing experiment + per-sample predictions)
        if predictions:
            bundle_path = run_dir / "bundle.json"
            predictions_json = []
            for row in predictions:
                row_out = dict(row)
                # Ensure the correctness flag exists in the JSON bundle too
                row_out["is_correct"] = self._is_correct(row.get("y_true"), row.get("y_pred"))
                predictions_json.append(self._convert_to_python(row_out))

            bundle = {
                "experiment": self._convert_to_python(experiment),
                "predictions": predictions_json,
            }
            with open(bundle_path, "w", encoding="utf-8") as f:
                json.dump(bundle, f, indent=2, ensure_ascii=False)
            experiment["artifacts"]["bundle_json"] = str(bundle_path)

        # Store the run directory as an artifact for convenience
        experiment["artifacts"]["run_dir"] = str(run_dir)
        
        self.history.append(experiment)
        self._save_history()
        
        print(f"\nOK: Logged experiment: {experiment_id}")
        print(f"OK: History file: {self.history_file}")
        print(f"OK: Run folder: {run_dir}")
        if experiment.get("artifacts", {}).get("predictions_csv"):
            print(f"OK: Predictions CSV: {experiment['artifacts']['predictions_csv']}")
        print(f"OK: Experiment JSON: {experiment['artifacts']['experiment_json']}")
        if experiment.get("artifacts", {}).get("bundle_json"):
            print(f"OK: Bundle JSON: {experiment['artifacts']['bundle_json']}")
        
        return experiment_id
    
    def get_best_experiment(self, metric: str = "qwk") -> Optional[Dict]:
        """
        Get the best experiment based on a metric.
        
        Args:
            metric: Metric to optimize ('qwk', 'accuracy', 'mse', 'mae')
        
        Returns:
            Best experiment dict or None
        """
        if not self.history:
            return None
        
        if metric in ['mse', 'mae']:
            # Lower is better
            return min(self.history, key=lambda x: x['metrics'][metric])
        else:
            # Higher is better
            return max(self.history, key=lambda x: x['metrics'][metric])
    
    def show_history(self, sort_by: str = "qwk"):
        """
        Display experiment history sorted by metric.
        
        Args:
            sort_by: Metric to sort by ('qwk', 'accuracy', 'mse', 'mae')
        """
        # Reload history to pick up experiments logged by other processes/cells
        self.history = self._load_history()
        
        if not self.history:
            print("\nNo experiments yet. Run an experiment first!")
            return
        
        print("\n" + "=" * 153)
        print(f"EXPERIMENT HISTORY - {self.category}")
        print("=" * 153)
        
        # Sort experiments
        reverse = sort_by not in ['mse', 'mae']  # Higher is better for qwk/accuracy
        sorted_history = sorted(
            self.history,
            key=lambda x: x['metrics'][sort_by],
            reverse=reverse
        )
        
        # Print header
        print(f"\n{'#':<4} {'Exp ID':<12} {'Model (HF ID)':<38} {'Template':<14} {'Prompt':<10} {'Quant':<8} {'Fmt':<6} "
              f"{'MaxTok':<8} {'Temp':<6} {'QWK':<8} {'Acc':<8} {'MSE':<8} {'Max Samples':<15} {'Valid Outputs':<15} {'Date':<20}")
        print("-" * 153)
        
        # Print experiments
        for rank, entry in enumerate(sorted_history, 1):
            marker = "  "
            date = entry['timestamp'][:19].replace('T', ' ')
            exp_id = entry.get('experiment_id', entry.get('version', 'N/A'))
            model_full = entry['model']  # Full HF ID
            # Shorten only if too long (keep first 36 chars to fit new column)
            model = model_full if len(model_full) <= 36 else model_full[:33] + "..."
            template = entry['prompt_template'][:12]
            prompt_hash = entry.get('prompt_hash', 'N/A')[:8]  # Show first 8 chars of hash
            quant = entry.get('quantization', 'none')[:6]
            output_fmt = entry.get('output_format', 'text')[:4]  # json or text
            qwk = entry['metrics']['qwk']
            acc = entry['metrics']['accuracy']
            mse = entry['metrics']['mse']
            samples_used = entry['dataset']['samples_used']
            max_samples = entry['dataset'].get('max_samples')
            total_samples = entry['dataset'].get('total_samples', samples_used)
            
            # Format max_samples display
            max_samples_str = "full test set" if max_samples is None else str(max_samples)
            
            # Format valid outputs display
            valid_outputs_str = f"{samples_used}/{total_samples}"
            
            # Get generation params (with fallback for old experiments)
            gen_params = entry.get('generation_params', {})
            max_tok = gen_params.get('max_new_tokens', 'N/A')
            temp = gen_params.get('temperature', 'N/A')
            max_tok_str = str(max_tok) if max_tok != 'N/A' else 'N/A'
            temp_str = f"{temp:.1f}" if temp != 'N/A' else 'N/A'
            
            print(f"{marker}{rank:<3} {exp_id:<12} {model:<38} {template:<14} {prompt_hash:<10} {quant:<8} {output_fmt:<6} "
                  f"{max_tok_str:<8} {temp_str:<6} {qwk:<8.4f} {acc:<8.2%} {mse:<8.4f} {max_samples_str:<15} {valid_outputs_str:<15} {date:<20}")
        
        print("=" * 153)
        best = sorted_history[0]
        best_model = best['model']
        best_template = best['prompt_template']
        best_output = best.get('output_format', 'text')
        best_qwk = best['metrics']['qwk']
        best_acc = best['metrics']['accuracy']
        best_mse = best['metrics']['mse']
        print(f"\n⭐ Best experiment for {self.category}: {best.get('experiment_id')} | Model: {best_model} | Template: {best_template} | Format: {best_output} | QWK: {best_qwk:.4f} | Acc: {best_acc:.2%} | MSE: {best_mse:.4f}")
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment by ID."""
        for entry in self.history:
            if entry.get('experiment_id') == experiment_id or entry.get('version') == experiment_id:
                return entry
        return None
    
    def show_experiment(self, experiment_id: str):
        """Display detailed information about an experiment."""
        exp = self.get_experiment(experiment_id)
        if not exp:
            print(f"\nERROR: Experiment '{experiment_id}' not found.")
            return
        
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {exp.get('experiment_id')}")
        print("=" * 80)
        print(f"\nTimestamp:   {exp['timestamp']}")
        print(f"Category:    {exp['category']}")
        print(f"Model:       {exp['model']}")
        print(f"Quantization: {exp.get('quantization', 'none')}")
        print(f"Output Format: {exp.get('output_format', 'text')}")
        print(f"Template:    {exp['prompt_template']}")
        print(f"Prompt Hash:   {exp.get('prompt_hash', 'N/A')} (first 8 chars of MD5)")
        
        # Display generation parameters
        if 'generation_params' in exp:
            print(f"\nGeneration Parameters:")
            gen_params = exp['generation_params']
            print(f"  max_new_tokens: {gen_params.get('max_new_tokens', 'N/A')}")
            print(f"  temperature:    {gen_params.get('temperature', 'N/A')}")
            print(f"  top_p:          {gen_params.get('top_p', 'N/A')}")
            print(f"  do_sample:      {gen_params.get('do_sample', 'N/A')}")
        
        print(f"\nDataset:")
        print(f"  Path:    {exp['dataset']['path']}")
        print(f"  Split:   {exp['dataset']['split']}")
        print(f"  Samples: {exp['dataset']['samples_used']}/{exp['dataset']['total_samples']}")
        
        print(f"\nMetrics:")
        for metric, value in exp['metrics'].items():
            print(f"  {metric:<10}: {value:.4f}")
        
        if exp.get('notes'):
            print(f"\nNotes: {exp['notes']}")
        
        print("\n" + "=" * 80)
    
    def compare_experiments(self, exp_id1: str, exp_id2: str):
        """Compare two experiments side by side."""
        exp1 = self.get_experiment(exp_id1)
        exp2 = self.get_experiment(exp_id2)
        
        if not exp1 or not exp2:
            print(f"\nERROR: Could not find both experiments.")
            return
        
        print("\n" + "=" * 80)
        print(f"COMPARING: {exp_id1} vs {exp_id2}")
        print("=" * 80)
        
        print(f"\n{'Metric':<15} {exp_id1:<20} {exp_id2:<20} {'Diff':<15}")
        print("-" * 80)
        
        for metric in ['accuracy', 'qwk', 'mse', 'mae']:
            val1 = exp1['metrics'][metric]
            val2 = exp2['metrics'][metric]
            diff = val2 - val1
            
            if metric in ['accuracy', 'qwk']:
                diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
                better = "OK" if diff > 0 else ""
            else:  # MSE, MAE - lower is better
                diff_str = f"{diff:.4f}"
                better = "OK" if diff < 0 else ""
            
            print(f"{metric:<15} {val1:<20.4f} {val2:<20.4f} {diff_str:<15} {better}")
        
        print("=" * 80)
    
    def export_best(self, output_file: Optional[str] = None):
        """Export best experiment prompt to file."""
        best = self.get_best_experiment()
        if not best:
            print("\nERROR: No experiments to export.")
            return
        
        if output_file is None:
            output_file = self.experiments_dir / f"{self.category}_best.txt"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Best Experiment for {self.category}\n")
            f.write(f"# Experiment ID: {best.get('experiment_id')}\n")
            f.write(f"# Model: {best['model']}\n")
            f.write(f"# Template: {best['prompt_template']}\n")
            f.write(f"# QWK: {best['metrics']['qwk']:.4f}\n")
            f.write(f"# Accuracy: {best['metrics']['accuracy']:.2%}\n")
            f.write(f"# Date: {best['timestamp']}\n\n")
            f.write(best['prompt_text'])
        
        print(f"\nOK: Best experiment exported to: {output_path}")
        print(f"  Experiment ID: {best.get('experiment_id')}")
        print(f"  QWK: {best['metrics']['qwk']:.4f}")
    
    def export_to_csv(self, sort_by: str = "qwk", output_path: Optional[str] = None) -> str:
        """
        Export experiment history for the chosen category to CSV in the same format as show_history.
                
        Args:
            sort_by: Metric to sort experiments by. Options:
                - 'qwk' (default): Sort by Quadratic Weighted Kappa (higher is better)
                - 'accuracy': Sort by accuracy (higher is better)
                - 'mse': Sort by Mean Squared Error (lower is better)
                - 'mae': Sort by Mean Absolute Error (lower is better)
            output_path: Path to save CSV. If None, saves to results/{category}_history.csv
        
        Returns:
            Path to the saved CSV file
        """        
        if not self.history:
            raise ValueError(f"No experiments found for category {self.category}")
        
        # Default output path
        if output_path is None:
            output_path = f"results/{self.category}_history.csv"
        
        # Sort experiments (same as show_history)
        reverse = sort_by not in ['mse', 'mae']  # Higher is better for qwk/accuracy
        sorted_history = sorted(
            self.history,
            key=lambda x: x['metrics'][sort_by],
            reverse=reverse
        )
        
        # Convert to DataFrame with exact same columns and formatting as show_history
        history_data = []
        for rank, entry in enumerate(sorted_history, 1):
            date = entry['timestamp'][:19].replace('T', ' ')
            exp_id = entry.get('experiment_id', entry.get('version', 'N/A'))
            model_full = entry['model']
            model = model_full if len(model_full) <= 36 else model_full[:33] + "..."
            template = entry['prompt_template'][:12]
            prompt_hash = entry.get('prompt_hash', 'N/A')[:8]
            quant = entry.get('quantization', 'none')[:6]
            output_fmt = entry.get('output_format', 'text')[:4]
            qwk = entry['metrics']['qwk']
            acc = entry['metrics']['accuracy']
            mse = entry['metrics']['mse']
            samples_used = entry['dataset']['samples_used']
            max_samples = entry['dataset'].get('max_samples')
            total_samples = entry['dataset'].get('total_samples', samples_used)
            
            # Format max_samples display
            max_samples_str = "full test set" if max_samples is None else str(max_samples)
            
            # Format valid outputs display
            valid_outputs_str = f"{samples_used}/{total_samples}"
            
            # Get generation params
            gen_params = entry.get('generation_params', {})
            max_tok = gen_params.get('max_new_tokens', 'N/A')
            temp = gen_params.get('temperature', 'N/A')
            max_tok_str = str(max_tok) if max_tok != 'N/A' else 'N/A'
            temp_str = f"{temp:.1f}" if temp != 'N/A' else 'N/A'
            
            row = {
                '#': rank,
                'Exp ID': exp_id,
                'Model (HF ID)': model,
                'Template': template,
                'Prompt': prompt_hash,
                'Quant': quant,
                'Fmt': output_fmt,
                'MaxTok': max_tok_str,
                'Temp': temp_str,
                'QWK': f"{qwk:.4f}",
                'Acc': f"{acc:.2%}",
                'MSE': f"{mse:.4f}",
                'Max Samples': max_samples_str,
                'Valid Outputs': valid_outputs_str,
                'Date': date,
            }
            history_data.append(row)
        
        # Create DataFrame and save
        history_df = pd.DataFrame(history_data)
        history_df.to_csv(output_path, index=False)
        
        return output_path
