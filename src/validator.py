"""
Dataset validation utilities.
"""

import pandas as pd
from constants import MINIMUM_ROWS, MINIMUM_FEATURES
import warnings


def _check_empty_dataset(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset is empty")


def _check_minimum_rows(df: pd.DataFrame) -> None:
    if len(df) < MINIMUM_ROWS:
        raise ValueError(
                f"Dataset must contain at least {MINIMUM_ROWS} rows."
            )


def _check_minimum_features(df: pd.DataFrame) -> None:
    if len(df.columns) < MINIMUM_FEATURES:
        raise ValueError(
                f"Dataset has fewer than {MINIMUM_FEATURES} features."
            )


def _check_duplicate_columns(df: pd.DataFrame) -> None:
    if df.columns.duplicated().any():
        raise ValueError("Dataset has duplicate column names")


def _check_missing_values(df: pd.DataFrame) -> None:
    if df.isnull().any().any():
        warnings.warn(
            "Dataset contains missing values. Preprocessing is recommended.",
            UserWarning
        )


def validate_dataset(df: pd.DataFrame) -> None:
    _check_empty_dataset(df)
    _check_duplicate_columns(df)
    _check_minimum_rows(df)
    _check_minimum_features(df)
    _check_missing_values(df)