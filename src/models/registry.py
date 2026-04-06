"""
Model registry with hardware requirements and recommendations.
Helps users select appropriate models for their hardware.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import torch
import logging

from ..utils.device import get_device, get_available_vram as get_vram

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    hf_id: str
    params: str  # e.g., "1.5B"
    params_b: float  # Numeric value in billions
    vram_gb: float  # Estimated VRAM without quantization
    vram_8bit_gb: float  # With 8-bit quantization
    vram_4bit_gb: float  # With 4-bit quantization
    speed: str  # "fast", "medium", "slow"
    quality: str  # "good", "excellent", "best"
    notes: str
    # Output format configuration based on empirical testing
    test_json: bool = False  # Test this model with JSON output format
    test_text: bool = True   # Test this model with text output format
    
    @property
    def output_format(self) -> str:
        """Get output format as string: 'json' or 'text'."""
        return "json" if self.test_json else "text"

# Model registry - all available models
MODELS = {
    ## excluded due to bad performance
    ## best run:   46  exp_042      Qwen3-0.6B             zero_shot      8bit     json     100      0.0    0.0384   47.30%   482        2026-01-16 00:13:37 
    # "qwen3-0.6b": ModelConfig(
    #     name="Qwen3 0.6B",
    #     hf_id="Qwen/Qwen3-0.6B",
    #     params="0.6B",
    #     params_b=0.6,
    #     vram_gb=1.2,
    #     vram_8bit_gb=0.6,
    #     vram_4bit_gb=0.3,
    #     speed="fast",
    #     quality="good",
    #     notes="11th BEST - Fastest option, ideal for quick tests and prompt development",
    #     test_json=False,
    #     test_text=True,
    #     json_max_tokens=50,
    # ),
    "qwen3-4b": ModelConfig(
        name="Qwen3 4B",
        hf_id="Qwen/Qwen3-4B",
        params="4B",
        params_b=4.0,
        vram_gb=8.0,
        vram_8bit_gb=4.0,
        vram_4bit_gb=2.0,
        speed="medium",
        quality="excellent",
        notes="🥉 3rd BEST - Excellent balance of speed and quality, latest Qwen3 architecture",
        test_json=False,
        test_text=True,
    ),
    ## excluded due to bad performance
    ## best run:   29  exp_009      Qwen3-8B               few_shot       8bit     text     N/A      N/A    0.2679   57.05%   482        2026-01-14 22:17:26 
    # "qwen3-8b": ModelConfig(
    #     name="Qwen3 8B",
    #     hf_id="Qwen/Qwen3-8B",
    #     params="8B",
    #     params_b=8.0,
    #     vram_gb=16.0,
    #     vram_8bit_gb=8.0,
    #     vram_4bit_gb=4.0,
    #     speed="slow",
    #     quality="best",
    #     notes="5th BEST - Latest Qwen3 model, requires 16GB+ GPU or quantization",
    #     test_json=False,
    #     test_text=True,
    #     json_max_tokens=50,
    # ),

    ## excluded due to bad performance
    ## best run:   37  exp_010      Qwen2.5-0.5B-Instruc   few_shot       8bit     text     N/A      N/A    0.1041   70.95%   482        2026-01-14 22:18:51 
    # "qwen2.5-0.5b": ModelConfig(
    #     name="Qwen2.5 0.5B",
    #     hf_id="Qwen/Qwen2.5-0.5B-Instruct",
    #     params="0.5B",
    #     params_b=0.5,
    #     vram_gb=1.0,
    #     vram_8bit_gb=0.6,
    #     vram_4bit_gb=0.4,
    #     speed="fast",
    #     quality="good",
    #     notes="8th BEST - Smallest model, works on any hardware, good for initial testing",
    #     test_json=False,
    #     test_text=True,
    #     json_max_tokens=50,
    # ),
    "qwen2.5-1.5b-text": ModelConfig(
        name="Qwen 2.5-1.5B (Text)",
        hf_id="Qwen/Qwen2.5-1.5B-Instruct",
        params="1.5B",
        params_b=1.5,
        vram_gb=3.0,
        vram_8bit_gb=1.6,
        vram_4bit_gb=1.0,
        speed="fast",
        quality="good",
        notes="🥈 2nd BEST - Text format (QWK: 0.3512)",
        test_json=False,
        test_text=True,
    ),
    "qwen2.5-1.5b-json": ModelConfig(
        name="Qwen 2.5-1.5B (JSON)",
        hf_id="Qwen/Qwen2.5-1.5B-Instruct",
        params="1.5B",
        params_b=1.5,
        vram_gb=3.0,
        vram_8bit_gb=1.6,
        vram_4bit_gb=1.0,
        speed="fast",
        quality="good",
        notes="🥈 2nd BEST - JSON format (QWK: 0.3053, text is better)",
        test_json=True,
        test_text=False,
    ),

    ## excluded due to bad performance
    ## best run:   28  exp_012      Qwen2.5-3B-Instruct    few_shot       8bit     text     N/A      N/A    0.2986   82.37%   482        2026-01-14 22:22:55 
    # "qwen2.5-3b": ModelConfig(
    #     name="Qwen2.5-3B",
    #     hf_id="Qwen/Qwen2.5-3B-Instruct",
    #     params="3B",
    #     params_b=3.0,
    #     vram_gb=6.0,
    #     vram_8bit_gb=3.2,
    #     vram_4bit_gb=2.0,
    #     speed="medium",
    #     quality="excellent",
    #     notes="4th BEST - Strong quality, fits 8GB GPUs with 8-bit quantization",
    #     test_json=False,
    #     test_text=True,
    #     json_max_tokens=50,
    # ),
    "qwen2.5-7b-json": ModelConfig(
        name="Qwen2.5-7B (JSON)",
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        params="7B",
        params_b=7.0,
        vram_gb=14.0,
        vram_8bit_gb=7.5,
        vram_4bit_gb=4.0,
        speed="slow",
        quality="best",
        notes="🏆 BEST PERFORMANCE - JSON format performs best",
        test_json=True,
        test_text=False,
    ),
    "qwen2.5-7b-text": ModelConfig(
        name="Qwen 2.5-7B (Text)",
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        params="7B",
        params_b=7.0,
        vram_gb=14.0,
        vram_8bit_gb=7.5,
        vram_4bit_gb=4.0,
        speed="slow",
        quality="best",
        notes="🏆 BEST PERFORMANCE - Best with JSON, but text format also strong",
        test_json=False,
        test_text=True,
    ),
    ## excluded model due to compatibility issues (loading so far not possible)
    # "apriel-5b": ModelConfig(
    #     name="Apriel 5B",
    #     hf_id="ServiceNow-AI/Apriel-5B-Instruct",
    #     params="5B",
    #     params_b=5.0,
    #     vram_gb=10.0,
    #     vram_8bit_gb=5.5,
    #     vram_4bit_gb=3.0,
    #     speed="medium",
    #     quality="excellent",
    #     notes="ServiceNow's instruction-tuned model, good balance of size and quality"
    # ),

## excluded due to bad performance
## best run:   36  exp_014      OLMo-2-1124-7B-Instr   few_shot       8bit     text     N/A      N/A    0.1203   42.41%   481        2026-01-14 22:27:29 
    # "olmo-7b": ModelConfig(
    #     name="OLMo 2 7B",
    #     hf_id="allenai/OLMo-2-1124-7B-Instruct",
    #     params="7B",
    #     params_b=7.0,
    #     vram_gb=14.0,
    #     vram_8bit_gb=7.0,
    #     vram_4bit_gb=3.5,
    #     speed="medium",
    #     quality="excellent",
    #     notes="7th BEST - Strong open-source model from AllenAI",
    #     test_json=False,
    #     test_text=True,
    #     json_max_tokens=50,
    # ),

    ## excluded due to bad performance. Many invalid answers.
    ## Best run:   33  exp_056      rnj-1                  zero_shot      8bit     json     100      0.0    0.2342   59.53%   472        2026-01-16 17:40:16 
    # "rnj-1": ModelConfig(
    #     name="RNJ-1",
    #     hf_id="EssentialAI/rnj-1",
    #     params="8B",
    #     params_b=8.0,  
    #     vram_gb=16.0,
    #     vram_8bit_gb=8.0,
    #     vram_4bit_gb=4.0,
    #     speed="medium",
    #     quality="good",
    #     notes="6th BEST - EssentialAI's 8B model. Best with JSON format.",
    #     test_json=True,  # JSON format performs best
    #     test_text=False,
    #     json_max_tokens=100,
    # ),

    ## excluded due to size
    # "nemotron-30b": ModelConfig(
    #     name="Nemotron 3 Nano 30B",
    #     hf_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
    #     params="30B",
    #     params_b=30.0,
    #     vram_gb=60.0,
    #     vram_8bit_gb=30.0,
    #     vram_4bit_gb=15.0,
    #     speed="slow",
    #     quality="best",
    #     notes="NVIDIA's large model with FP8 quantization, highest quality but requires significant VRAM (40GB+ recommended)",
    # ),

    ## exclueded due to bad performance, a lot of invalid answers
    ## best run:   39  exp_016      DeepSeek-R1-0528-Qwe   few_shot       8bit     text     N/A      N/A    0.0981   36.19%   420        2026-01-14 22:37:47 
    # "deepseek-r1-qwen3-8b": ModelConfig(
    #     name="DeepSeek R1 Qwen3 8B",
    #     hf_id="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    #     params="8B",
    #     params_b=8.0,
    #     vram_gb=16.0,
    #     vram_8bit_gb=8.0,
    #     vram_4bit_gb=4.0,
    #     speed="medium",
    #     quality="best",
    #     notes="9th BEST - DeepSeek's reasoning model with Qwen3 architecture",
    #     test_json=False,
    #     test_text=True,
    #     json_max_tokens=50,
    # ),

    ## excluded due to missing authorization by author on huggingface
    # "llama3.1-8b": ModelConfig(
    #     name="Llama 3.1 8B",
    #     hf_id="meta-llama/Llama-3.1-8B-Instruct",
    #     params="8B",
    #     params_b=8.0,
    #     vram_gb=16.0,
    #     vram_8bit_gb=8.0,
    #     vram_4bit_gb=4.0,
    #     speed="slow",
    #     quality="best",
    #     notes="Meta's latest model, excellent reasoning (requires HF access)"
    # ),

    ## excluded due to bad performance
    ## best run:   40  exp_047      Llama-3.2-3B-Instruc   zero_shot      8bit     json     100      0.0    0.0868   47.93%   482        2026-01-16 04:18:19 
    # "llama3.2-3b": ModelConfig(
    #     name="Llama 3.2 3B",
    #     hf_id="meta-llama/Llama-3.2-3B-Instruct",
    #     params="3B",
    #     params_b=3.0,
    #     vram_gb=6.0,
    #     vram_8bit_gb=3.0,
    #     vram_4bit_gb=1.5,
    #     speed="fast",
    #     quality="good",
    #     notes="10th BEST - Lightweight Llama, efficient for limited VRAM (requires HF access). Best with JSON format.",
    #     test_json=True,  # JSON format performs best
    #     test_text=False,
    #     json_max_tokens=100,
    # ),
}


def get_compatible_models(
    use_8bit: bool = False,
    safety_margin_gb: float = 2.0
) -> List[str]:
    """
    Get list of models compatible with current hardware.
    
    Args:
        use_8bit: Whether 8-bit quantization will be used
        safety_margin_gb: Reserve this much VRAM for other operations
    
    Returns:
        List of compatible model keys
    """
    device = get_device()
    available_vram = get_vram()
    
    if device.type == "cpu":
        # CPU mode - all models work but slower
        logger.warning("No GPU detected - all models will run on CPU (slower)")
        return list(MODELS.keys())
    elif device.type == "mps":
        # MPS (Apple Silicon) - unified memory, all models compatible
        logger.info("Apple Silicon detected - unified memory (all models compatible)")
        return list(MODELS.keys())
    
    # CUDA - check VRAM
    if available_vram is None:
        logger.warning("Could not determine VRAM - returning all models")
        return list(MODELS.keys())
    
    logger.info(f"Available GPU VRAM: {available_vram:.1f}GB")
    
    compatible = []
    
    for key, model in MODELS.items():
        if use_8bit:
            required_vram = model.vram_8bit_gb
        else:
            required_vram = model.vram_gb
        
        if required_vram + safety_margin_gb <= available_vram:
            compatible.append(key)
            logger.info(f"✓ {model.name} compatible ({required_vram:.1f}GB needed)")
        else:
            logger.warning(f"✗ {model.name} requires {required_vram:.1f}GB (not enough VRAM)")
    
    return compatible


def print_model_info():
    """Print formatted table of all models."""
    print("\n" + "=" * 110)
    print("AVAILABLE MODELS FOR DIKOBI CLASSIFICATION")
    print("=" * 110)
    print(f"{'Key':<15} {'Name':<20} {'Params':<8} {'VRAM':<10} {'8-bit':<10} {'4-bit':<10} {'Speed':<10} {'Quality':<10}")
    print("-" * 110)
    
    for key, model in MODELS.items():
        print(f"{key:<15} {model.name:<20} {model.params:<8} {model.vram_gb:<10.1f} "
              f"{model.vram_8bit_gb:<10.1f} {model.vram_4bit_gb:<10.1f} {model.speed:<10} {model.quality:<10}")
    
    print("-" * 110)
    print("\nRecommendations:")
    for key, model in MODELS.items():
        marker = "⭐" if "RECOMMENDED" in model.notes else "  "
        print(f"{marker} {model.name}: {model.notes}")
    print("=" * 110 + "\n")


def get_model_config(model_key: str) -> ModelConfig:
    """
    Get configuration for a specific model.
    
    Args:
        model_key: Model key from MODELS dict
    
    Returns:
        ModelConfig object
    
    Raises:
        ValueError: If model_key is unknown
    """
    if model_key not in MODELS:
        raise ValueError(
            f"Unknown model: {model_key}. "
            f"Available models: {', '.join(MODELS.keys())}"
        )
    return MODELS[model_key]


def get_recommended_model() -> str:
    """
    Get recommended model based on available hardware.
    
    Returns:
        Model key for recommended model
    """
    available_vram = get_available_vram()
    
    if available_vram is None:
        # CPU mode - use smallest model
        return "qwen2.5-0.5b"
    elif available_vram < 4:
        # Very limited VRAM
        return "qwen2.5-0.5b"
    elif available_vram < 8:
        # ~4-8GB VRAM
        return "qwen2.5-1.5b"
    elif available_vram < 16:
        # ~8-16GB VRAM
        return "qwen2.5-3b"
    else:
        # 16GB+ VRAM - use best quality
        return "qwen2.5-7b"
