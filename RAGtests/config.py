"""
Configuration for few-shot prompting experiments.
Focus: Small models (Qwen2.5 3B), MSE optimization, few-shot learning
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# Paths
TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = TESTS_DIR / "results"
EXPERIMENTS_DIR = TESTS_DIR / "experiments"

# Model Configuration
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MODEL_OPTIONS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct"
]

# Few-Shot Configuration
DEFAULT_NUM_SHOTS = [1, 3, 5, 7]  # Test different numbers of examples
MAX_SHOTS = 10  # Maximum examples to test

# Example Selection Strategies
SELECTION_STRATEGIES = [
    "random",           # Random selection from training set
    "balanced",         # Equal distribution across rating classes
    "stratified",       # Stratified by rating distribution
    "diverse",          # Maximize text diversity (length, complexity)
    "representative",   # Most typical examples per class
    "boundary",         # Examples near decision boundaries
    "hard",             # Most difficult examples (high confusion)
    "easy"              # Clearest examples (low confusion)
]

# Prompt Template Variations
PROMPT_FORMATS = [
    "standard",         # Basic few-shot with examples
    "explicit_scale",   # Emphasize rating scale explanation
    "step_by_step",     # Chain-of-thought reasoning
    "criteria_focused", # Highlight evaluation criteria
    "comparative"       # Compare to examples explicitly
]

# Evaluation Configuration
PRIMARY_METRIC = "mse"  # Mean Squared Error (primary focus)
SECONDARY_METRICS = ["mae", "accuracy", "qwk"]  # Additional metrics

# Test Configuration
DEFAULT_TEST_SIZE = 50  # Number of test samples (None = full test set)
RANDOM_SEED = 42
USE_8BIT = True  # 8-bit quantization for memory efficiency
BATCH_SIZE = 2  # Small batch size for stability

# Generation Parameters (optimized for small models)
GENERATION_CONFIG = {
    "max_new_tokens": 10,
    "temperature": 0.1,  # Low temperature for consistency
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1
}


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    category: str
    model: str = DEFAULT_MODEL
    num_shots: List[int] = None
    selection_strategy: str = "balanced"
    prompt_format: str = "standard"
    test_size: Optional[int] = DEFAULT_TEST_SIZE
    use_8bit: bool = USE_8BIT
    batch_size: int = BATCH_SIZE
    random_seed: int = RANDOM_SEED
    
    def __post_init__(self):
        if self.num_shots is None:
            self.num_shots = DEFAULT_NUM_SHOTS
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "category": self.category,
            "model": self.model,
            "num_shots": self.num_shots,
            "selection_strategy": self.selection_strategy,
            "prompt_format": self.prompt_format,
            "test_size": self.test_size,
            "use_8bit": self.use_8bit,
            "batch_size": self.batch_size,
            "random_seed": self.random_seed
        }


# Quick Test Categories (start with these)
QUICK_TEST_CATEGORIES = [
    "1_D_M",      # Case 1: Description - Motivation
    "1_D_kA",     # Case 1: Description - Cognitive Activation
    "3_D_F",      # Case 3: Description - Technical Language
]

# All available categories
ALL_CATEGORIES = [
    "1_D_M", "1_D_kA", "1_Dm_kA", "1_Dm_M",
    "3_D_F", "3_D_Qual", "3_D_Quan", 
    "3_Dm_F", "3_Dm_Kons", "3_Dm_Qual", "3_Dm_Quan",
    "4_D_Ep", "4_Dm_Ep",
    "5_D_Ma", "5_D_Mk", "5_Dm_Ma", "5_Dm_Mk",
    "6_D_kA", "6_D_Rb", "6_Dm_kA", "6_Dm_Rb",
    "1_E_T_IV", "3_E_T_IV", "4_E_T_IV", "5_E_T_IV", "6_E_T_IV"
]
