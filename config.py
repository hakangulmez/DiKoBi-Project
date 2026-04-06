"""
Configuration file for diKobi LLM Classification System - Project paths only

Used by preprocessing scripts to locate data directories.
All other settings (model, category, tokens, etc.) are configured in the notebook.
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS (for preprocessing scripts)
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"