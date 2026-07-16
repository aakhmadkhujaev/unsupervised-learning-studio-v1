"""
Configuration settings for Unsupervised Learning Studio v1.0.

Contains project paths and global application settings.
"""

from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# ==========================================================
# Global Settings
# ==========================================================

RANDOM_STATE = 42

FIGURE_DPI = 300

FIGURE_SIZE: tuple[int, int] = (10, 6)