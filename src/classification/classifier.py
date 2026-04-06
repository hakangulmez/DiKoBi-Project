"""
Text classifier using local LLMs for DiKoBi categories.
Simplified interface focused on classification tasks.
"""

import re
import logging
from typing import Optional, Dict, Any, List
import torch
from transformers import pipeline

from ..models import load_model, unload_model
from ..utils.device import get_device
from .prompts import get_prompt, get_valid_outputs

logger = logging.getLogger(__name__)


class TextClassifier:
    """
    LLM-based text classifier for DiKoBi categories.
    
    Simplified interface that handles:
    - Automatic model loading
    - Single text or batch processing
    - Score extraction from LLM outputs
    - Memory management
    
    Example:
        classifier = TextClassifier(model_name="qwen-1.5b")
        result = classifier.classify(category="1_D_M", text="...")
        
        # Or batch processing
        results = classifier.classify_batch(
            category="1_D_M",
            texts=["text1", "text2", ...]
        )
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        use_8bit: bool = None,  # None = auto-detect
        max_new_tokens: int = None,  # None = auto from registry
        temperature: float = 0.0,  # Greedy decoding for speed
        top_p: float = 1.0,  # Nucleus sampling (1.0 = disabled)
        do_sample: bool = False,  # Use sampling (False = greedy)
        batch_size: int = None,  # None = auto-detect
    ):
        """
        Initialize the classifier with automatic hardware optimization.
        
        Args:
            model_name: HuggingFace model ID or key from model registry
            use_8bit: Use 8-bit quantization (None = auto-detect based on hardware)
            max_new_tokens: Maximum tokens to generate (None = auto from registry)
            temperature: Sampling temperature (0.0 = deterministic/greedy)
            top_p: Nucleus sampling threshold (1.0 = disabled)
            do_sample: Enable sampling (False = greedy decoding)
            batch_size: Number of samples to process at once (None = auto-detect)
        """
        self.model_name = model_name
        
        # Try to load settings from registry
        model_config = None
        from ..models import MODELS
        
        # Check if model_name is a key or HF ID in registry
        for key, config in MODELS.items():
            if config.hf_id == model_name or key == model_name:
                model_config = config
                break
        
        # Apply settings from registry if available
        if model_config:
            # Each registry entry defines exactly ONE format
            use_json = model_config.test_json
            logger.info(f"Using format for {model_config.name}: {'JSON' if use_json else 'Text'}")
            
            # Set tokens based on format - require explicit configuration
            if max_new_tokens is None:
                try:
                    import __main__
                    if use_json:
                        max_new_tokens = getattr(__main__, 'JSON_MAX_TOKENS', None)
                        if max_new_tokens is None:
                            raise ValueError(
                                f"JSON_MAX_TOKENS not defined in notebook configuration.\n"
                                f"Model '{model_config.name}' uses JSON output format.\n"
                                f"Add 'JSON_MAX_TOKENS = <value>' to your notebook config cell,\n"
                                f"or pass max_new_tokens explicitly to TextClassifier()."
                            )
                    else:
                        max_new_tokens = getattr(__main__, 'TEXT_MAX_TOKENS', None)
                        if max_new_tokens is None:
                            raise ValueError(
                                f"TEXT_MAX_TOKENS not defined in notebook configuration.\n"
                                f"Model '{model_config.name}' uses Text output format.\n"
                                f"Add 'TEXT_MAX_TOKENS = <value>' to your notebook config cell,\n"
                                f"or pass max_new_tokens explicitly to TextClassifier()."
                            )
                except AttributeError:
                    raise ValueError(
                        f"Cannot access notebook configuration.\n"
                        f"Add 'TEXT_MAX_TOKENS' or 'JSON_MAX_TOKENS' to your notebook config cell,\n"
                        f"or pass max_new_tokens explicitly to TextClassifier()."
                    )
                logger.info(f"Using max_tokens for {model_config.name}: {max_new_tokens}")
            
            # Use HuggingFace ID from registry
            hf_model_id = model_config.hf_id
        else:
            # Fallback to defaults if model not in registry
            use_json = False  # Default to text (faster)
            if max_new_tokens is None:
                max_new_tokens = TEXT_MAX_TOKENS
        
        self.use_json = use_json
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        
        # Detect device
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Auto-detect quantization if not specified
        if use_8bit is None:
            use_8bit = self._auto_detect_quantization(hf_model_id)
        
        self.use_8bit = use_8bit or False
        
        # Auto-detect batch size if not specified
        if batch_size is None:
            batch_size = self._auto_detect_batch_size()
        
        self.batch_size = batch_size
        logger.info(f"Batch size: {batch_size}")
        
        # Load model using HuggingFace ID
        logger.info(f"Loading model: {hf_model_id}")
        self.model, self.tokenizer = load_model(
            model_name=hf_model_id,
            use_8bit=use_8bit
        )
        
        # Create text generation pipeline. When the model is loaded with device_map="auto"
        # (accelerate), passing a device breaks. Use device_map on CUDA; use device elsewhere.
        device_type = self.device.type  # Extract 'cuda', 'mps', or 'cpu' from device object
        device_arg = 0 if device_type == "cuda" else -1 if device_type == "cpu" else "mps"

        pipeline_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
        }

        if device_type == "cuda":
            pipeline_kwargs["device_map"] = "auto"
        else:
            pipeline_kwargs["device"] = device_arg

        self.pipeline = pipeline("text-generation", **pipeline_kwargs)
        
        logger.info("✓ Classifier ready")
    
    def get_generation_params(self) -> Dict[str, Any]:
        """Return the generation parameters used by this classifier."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "batch_size": self.batch_size,
        }
    
    def _auto_detect_quantization(self, model_name: str) -> bool:
        """
        Automatically determine if 8-bit quantization is needed based on model size and hardware.
        
        Returns:
            True if 8-bit quantization should be used, False otherwise
        """
        from ..utils.device import get_available_vram, recommend_quantization
        from ..models import get_model_config, MODELS
        
        # Try to get model config
        try:
            # Check if it's a key in our registry
            for key, config in MODELS.items():
                if config.hf_id == model_name or key == model_name:
                    params_b = config.params_b
                    break
            else:
                # Guess from model name
                if "0.5B" in model_name or "500M" in model_name:
                    params_b = 0.5
                elif "1.1B" in model_name or "1B" in model_name:
                    params_b = 1.1
                elif "1.5B" in model_name:
                    params_b = 1.5
                elif "2.7B" in model_name or "3B" in model_name:
                    params_b = 3.0
                elif "7B" in model_name:
                    params_b = 7.0
                else:
                    logger.warning(f"Could not determine model size, assuming no quantization needed")
                    return False, False
            
            use_8bit, message = recommend_quantization(params_b)
            logger.info(f"Auto-quantization: {message}")
            return use_8bit
            
        except Exception as e:
            logger.warning(f"Could not auto-detect quantization: {e}")
            return False
    
    def _auto_detect_batch_size(self) -> int:
        """
        Automatically determine optimal batch size based on hardware.
        
        Returns:
            Recommended batch size
        """
        from ..utils.device import get_available_vram
        
        if self.device.type == "cpu":
            return 1  # CPU can only do 1 at a time efficiently
        
        if self.device.type == "mps":
            return 1  # MPS is slow with batching, use 1
        
        # CUDA - check VRAM
        vram = get_available_vram()
        if vram is None:
            return 4
        
        if vram < 4:
            return 2
        elif vram < 8:
            return 4
        elif vram < 16:
            return 8
        else:
            return 16
    
    def _extract_score(self, response: str, valid_outputs: list) -> Optional[int]:
        """
        Extract numerical score from model response.
        Handles both text-based ("Rating: X") and JSON formats.
        
        Args:
            response: Raw model output
            valid_outputs: List of valid scores
            
        Returns:
            Extracted score or None if invalid
        """
        if not response:
            return None
        
        response = response.strip()
        
        # 1) Try parsing as JSON first (if JSON mode is enabled)
        if self.use_json or '{' in response:
            try:
                import json
                # Try to find JSON object in response
                json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                if json_match:
                    json_obj = json.loads(json_match.group(0))
                    # Look for rating field (case-insensitive)
                    for key in ['rating', 'Rating', 'RATING', 'score', 'Score']:
                        if key in json_obj:
                            num = int(json_obj[key])
                            if num in valid_outputs:
                                return num
            except (json.JSONDecodeError, ValueError, KeyError):
                pass  # Fall through to text-based parsing
        
        # 2) Try the whole response as a single digit
        if response.isdigit():
            num = int(response)
            if num in valid_outputs:
                return num
        
        # 3) Try first line only (model often answers first)
        first_line = response.split("\n")[0].strip()
        if first_line.isdigit():
            num = int(first_line)
            if num in valid_outputs:
                return num
        
        # 4) Try first token with punctuation stripped
        first_token = response.split()[0].strip(".,!?()[]{}:;\"'") if response.split() else ""
        if first_token.isdigit():
            num = int(first_token)
            if num in valid_outputs:
                return num
        
        # 5) Look for explicit rating/score patterns
        explicit_patterns = [
            r"\b(?:rating|score|bewertung|punkte|punktzahl)\b\s*[:=\-]?\s*([0-9]+)",
            r"\"(?:rating|score)\"\s*:\s*([0-9]+)",  # JSON field
        ]
        for pat in explicit_patterns:
            m = re.search(pat, response, flags=re.IGNORECASE)
            if m:
                try:
                    num = int(m.group(1))
                except Exception:
                    num = None
                if num in valid_outputs:
                    return num

        # 6) Fallback: find any standalone integer and pick first valid one
        numbers = re.findall(r"\b\d+\b", response)
        for num_str in numbers:
            num = int(num_str)
            if num in valid_outputs:
                return num

        # If no valid number found, return None
        logger.warning(f"Could not extract valid score from: {response[:100]!r}")
        return None


    def classify(
        self,
        category: str,
        text: str,
        prompt_template: str = "zero_shot",
        return_raw: bool = False
    ) -> Dict[str, Any]:
        """
        Classify a single text for a specific category.
        
        Args:
            category: DiKoBi category (e.g., '1_D_M')
            text: Input text to classify
            prompt_template: Prompt variation to use (default: "zero_shot")
            return_raw: Whether to include raw model output
            
        Returns:
            Dictionary with:
                - score: Extracted numerical score (or None)
                - category: Category name
                - valid_range: Valid output range
                - raw_output: Raw model response (if return_raw=True)
        """
        # Get prompt and valid outputs
        output_format = "json" if self.use_json else "text"
        prompt = get_prompt(category, text, prompt_template, output_format=output_format)
        valid_outputs = get_valid_outputs(category)
        
        # Generate response
        # Only pass sampling parameters when actually sampling (do_sample=True)
        # to avoid warnings during greedy decoding
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "return_full_text": False,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        
        with torch.no_grad():
            outputs = self.pipeline(prompt, **gen_kwargs)
        
        raw_output = outputs[0]["generated_text"]
        score = self._extract_score(raw_output, valid_outputs)
        
        result = {
            "score": score,
            "category": category,
            "valid_range": valid_outputs,
        }
        
        if return_raw:
            result["raw_output"] = raw_output
        
        return result
    
    def classify_batch(
        self,
        category: str,
        texts: List[str],
        prompt_template: str = "zero_shot",
        return_raw: bool = False,
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple texts for a specific category.
        
        Args:
            category: DiKoBi category
            texts: List of input texts
            prompt_template: Prompt variation to use
            return_raw: Whether to include raw model outputs
            show_progress: Show progress bar
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        # Get valid outputs once
        valid_outputs = get_valid_outputs(category)
        
        # Setup progress tracking for batches
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        if show_progress:
            try:
                from tqdm import tqdm
                batch_iter = tqdm(
                    range(0, len(texts), self.batch_size),
                    desc=f"Classifying {category}",
                    total=num_batches,
                    unit="batch"
                )
            except ImportError:
                batch_iter = range(0, len(texts), self.batch_size)
                logger.info(f"Processing {len(texts)} texts in {num_batches} batches...")
        else:
            batch_iter = range(0, len(texts), self.batch_size)
        
        # Create prompts list for batching
        output_format = "json" if self.use_json else "text"
        prompts = [get_prompt(category, text, prompt_template, output_format=output_format) for text in texts]
        
        # Process in batches
        # Only pass sampling parameters when actually sampling (do_sample=True)
        # to avoid warnings during greedy decoding
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "return_full_text": False,
            "batch_size": self.batch_size,
        }
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        
        with torch.no_grad():
            outputs = self.pipeline(prompts, **gen_kwargs)
        
        # Process all outputs
        for i, output in enumerate(outputs):
            # If output is a list, take first element
            if isinstance(output, list):
                output = output[0]
                
            raw_output = output["generated_text"]
            score = self._extract_score(raw_output, valid_outputs)
            
            result = {
                "score": score,
                "category": category,
                "valid_range": valid_outputs,
                "parsing_failed": score is None,  # Track if parsing failed
            }
            
            # Always include raw output if parsing failed (for debugging)
            if return_raw or score is None:
                result["raw_output"] = raw_output
            
            results.append(result)
            
            # Update progress if using tqdm
            if show_progress and hasattr(batch_iter, 'update'):
                if (i + 1) % self.batch_size == 0 or i == len(texts) - 1:
                    batch_iter.update(1)
        
        return results
    
    def get_generation_params(self) -> Dict[str, Any]:
        """
        Get current generation parameters for experiment tracking.
        
        Returns:
            Dict with generation parameters including:
                - max_new_tokens
                - temperature
                - top_p
                - do_sample
                - batch_size
        """
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "batch_size": self.batch_size,
        }
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        import gc
        
        # Force garbage collection first
        gc.collect()
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
            # Additional aggressive cleanup for MPS
            gc.collect()  # Second pass after cache clear
    
    def unload(self):
        """Unload model and free all resources."""
        logger.info("Unloading classifier...")
        
        # Delete pipeline first (contains references to model/tokenizer)
        if hasattr(self, 'pipeline'):
            del self.pipeline
        
        # Delete model and tokenizer
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        
        # Call global unload
        unload_model()
        
        # Clear cache
        self.clear_cache()
        
        # Force garbage collection (multiple passes for MPS)
        import gc
        gc.collect()
        
        # Additional MPS-specific cleanup
        if self.device.type == "mps":
            torch.mps.empty_cache()
            gc.collect()  # Final pass
        
        logger.info("✓ Classifier unloaded")
