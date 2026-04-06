"""
Model loading utilities.
Handles loading and unloading models with appropriate device settings.
"""
import os
import torch
import logging
from typing import Optional, Dict, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from ..utils.device import get_device, get_device_info, get_available_vram

logger = logging.getLogger(__name__)

# Global model cache
_current_model = None
_current_tokenizer = None
_current_model_name = None


def load_model(
    model_name: str,
    use_8bit: bool = False,
    force_reload: bool = False,
    hf_token: Optional[str] = None, 
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer with appropriate device settings.
    
    Args:
        model_name: HuggingFace model identifier
        use_8bit: Use 8-bit quantization
        force_reload: Force reload even if model is cached
        hf_token: HuggingFace token for gated models. If None, automatically reads from HF_TOKEN environment variable.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    global _current_model, _current_tokenizer, _current_model_name
    
    # Get token from environment if not provided
    if hf_token is None:
        hf_token = os.getenv("HF_TOKEN")
    
    # Return cached model if available
    if not force_reload and _current_model is not None and _current_model_name == model_name:
        logger.info(f"Using cached model: {model_name}")
        return _current_model, _current_tokenizer
    
    # Unload previous model before loading new one
    if _current_model is not None:
        logger.info("Unloading previous model before loading new one...")
        unload_model()
    
    logger.info(f"Loading model: {model_name}")
    
    # Detect device
    device = get_device()
    logger.info(f"Device: {device}")
    
    # Setup quantization config
    quantization_config = None
    if use_8bit and device.type == "cuda":
        logger.info("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use left padding for decoder-only models to avoid right-padding warning
    tokenizer.padding_side = "left"
    
    # Load model with appropriate settings
    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    
    if device.type == "cuda":
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
    elif device.type == "mps":
        model_kwargs["dtype"] = torch.float16
    else:
        model_kwargs["dtype"] = torch.float32
    
    logger.info("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
        token=hf_token
    )
    
    # Move to device if not using device_map
    if "device_map" not in model_kwargs:
        model = model.to(device)
    
    model.eval()
    
    # Log memory usage
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    # Cache model
    _current_model = model
    _current_tokenizer = tokenizer
    _current_model_name = model_name
    
    logger.info("✓ Model loaded successfully")
    
    return model, tokenizer


def unload_model():
    """Unload current model and free memory."""
    global _current_model, _current_tokenizer, _current_model_name
    
    if _current_model is None:
        return
    
    logger.info(f"Unloading model: {_current_model_name}")
    
    device = get_device()
    
    # Delete model and tokenizer
    del _current_model
    del _current_tokenizer
    
    # Reset globals
    _current_model = None
    _current_tokenizer = None
    _current_model_name = None
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear device cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Log memory after cleanup
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            logger.info(f"GPU memory after cleanup: {allocated:.2f}GB allocated")
    elif device.type == "mps":
        torch.mps.empty_cache()
    
    logger.info("✓ Model unloaded")


def get_current_model_info() -> Optional[Dict]:
    """
    Get information about currently loaded model.
    
    Returns:
        Dictionary with model info, or None if no model loaded
    """
    if _current_model is None:
        return None
    
    device = get_device()
    info = {
        "model_name": _current_model_name,
        "device": device.type,
    }
    
    if device.type == "cuda":
        info["allocated_vram_gb"] = torch.cuda.memory_allocated(0) / 1024**3
        info["reserved_vram_gb"] = torch.cuda.memory_reserved(0) / 1024**3
    
    return info


def print_current_model_info():
    """Print information about currently loaded model."""
    info = get_current_model_info()
    
    if info is None:
        print("No model currently loaded.")
        return
    
    print("\n" + "=" * 60)
    print("CURRENT MODEL")
    print("=" * 60)
    print(f"Model: {info['model_name']}")
    print(f"Device: {info['device']}")
    
    if "allocated_vram_gb" in info:
        print(f"GPU Memory: {info['allocated_vram_gb']:.2f}GB allocated")
        print(f"            {info['reserved_vram_gb']:.2f}GB reserved")
    
    print("=" * 60 + "\n")
