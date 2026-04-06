"""
Device detection and hardware information utilities.
Handles GPU/CPU detection for all platforms (NVIDIA, Apple Silicon, CPU-only).
"""

import torch
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Get the best available device as a torch.device object.
    
    Returns:
        torch.device object ready for model.to(device)
    
    Example:
        device = get_device()
        model.to(device)
        
        # To get device type as string:
        device_str = device.type  # 'cuda', 'mps', or 'cpu'
    """
    return torch.device(
        "cuda:0" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() else 
        "cpu"
    )


def get_available_vram() -> Optional[float]:
    """
    Get available GPU VRAM in GB.
    
    Returns:
        Available VRAM in GB, or None if no GPU
    """
    device = get_device()
    
    if device.type == "cuda":
        free_memory, total_memory = torch.cuda.mem_get_info(0)
        return free_memory / 1024**3
    elif device.type == "mps":
        # MPS doesn't provide memory info, estimate based on unified memory
        # M1/M2/M3 typically have 8-32GB unified memory
        return None  # Can't reliably determine on MPS
    else:
        return None


def get_total_vram() -> Optional[float]:
    """
    Get total GPU VRAM in GB.
    
    Returns:
        Total VRAM in GB, or None if no GPU
    """
    device = get_device()
    
    if device.type == "cuda":
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        return None


def get_device_info() -> Dict[str, any]:
    """
    Get comprehensive device information.
    
    Returns:
        Dictionary with device details
    """
    device = get_device()
    info = {"device": device.type}
    
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["total_vram_gb"] = get_total_vram()
        info["available_vram_gb"] = get_available_vram()
        info["cuda_version"] = torch.version.cuda
    elif device.type == "mps":
        info["gpu_name"] = "Apple Silicon (MPS)"
        info["note"] = "Unified memory - no separate VRAM"
    else:
        info["note"] = "CPU-only mode (slower but works)"
    
    return info


def print_device_info():
    """Print formatted device information to console."""
    info = get_device_info()
    
    print("\n" + "=" * 60)
    print("HARDWARE INFORMATION")
    print("=" * 60)
    print(f"Device: {info['device'].upper()}")
    
    if info['device'] == "cuda":
        print(f"GPU: {info['gpu_name']}")
        print(f"Total VRAM: {info['total_vram_gb']:.1f} GB")
        print(f"Available VRAM: {info['available_vram_gb']:.1f} GB")
        print(f"CUDA Version: {info['cuda_version']}")
    elif info['device'] == "mps":
        print(f"GPU: {info['gpu_name']}")
        print(f"Note: {info['note']}")
    else:
        print(f"Note: {info['note']}")
    
    print("=" * 60 + "\n")


def log_memory_usage(context: str = ""):
    """Log current memory usage."""
    device = get_device()
    
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"{context} - GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    elif device.type == "mps":
        logger.info(f"{context} - MPS (unified memory)")
    else:
        logger.info(f"{context} - CPU mode")


def recommend_quantization(model_params_b: float) -> Tuple[bool, str]:
    """
    Recommend quantization settings based on hardware.
    
    Args:
        model_params_b: Model size in billions of parameters
    
    Returns:
        Tuple of (use_8bit, recommendation_message)
    """
    device = get_device()
    
    if device.type == "cpu":
        return False, "CPU mode - quantization not needed"
    
    if device.type == "mps":
        return False, "MPS mode - quantization not supported on Apple Silicon"
    
    # CUDA recommendations
    available_vram = get_available_vram()
    if available_vram is None:
        return False, "Could not determine VRAM"
    
    # Rough estimate: model needs ~2GB per billion parameters
    estimated_vram_needed = model_params_b * 2
    
    if estimated_vram_needed <= available_vram:
        return False, f"No quantization needed ({available_vram:.1f}GB available)"
    else:
        return True, f"8-bit quantization recommended ({available_vram:.1f}GB available)"
