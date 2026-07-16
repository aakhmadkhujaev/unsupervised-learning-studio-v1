"""
Dataset loading utilities for Unsupervised Learning Studio v1.0.
"""

from pathlib import Path

import pandas as pd


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """
    Load a dataset from a CSV file.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")

    try:
        return pd.read_csv(path)

    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty.")