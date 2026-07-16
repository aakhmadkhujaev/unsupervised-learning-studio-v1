

"""
Utility functions for Unsupervised Learning Studio v1.0.
"""

# imports
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from config import (
    ARTIFACTS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    FIGURE_DPI
)

# ==========================
# Console Helpers
# ==========================

def print_header(title: str) -> None:
    """
    Prints a formatted header for the console output.

    Args:
        title (str): The title to be printed as a header.
    """
    print("\n" + "=" * 50)
    print(f"{title.center(50)}")
    print("=" * 50 + "\n")

def print_section(title: str) -> None:
    """
    Prints a formatted section title for the console output.

    Args:
        title (str): The title to be printed as a section header.
    """
    print(f"\n{title}")
    print("-" * len(title))

# ==========================
# Directory Helpers
# ==========================

def create_output_directories() -> None:
    """
    Creates necessary output directories for reports and visualizations.
    """
    directories = [ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory created or already exists: {directory}")

# ==========================
# Time Helpers
# ==========================

def get_timestamp() -> str:
    """
    Returns the current timestamp in a formatted string.

    Returns:
        str: The current timestamp in "%Y%m%d_%H%M%S" format.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ==========================
# Visualization Helpers
# ==========================

def save_figure(filename: str) -> Path:
    """
    Saves a matplotlib figure to the specified filename.

    Args:
       filename (str): Base filename for the saved figure.
    """

    filename = f"{filename}_{get_timestamp()}.png"
    figure_path = Path(FIGURES_DIR) / filename
    plt.savefig(figure_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    return figure_path