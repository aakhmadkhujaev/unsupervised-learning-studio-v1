"""
Data preprocessing utilities for Unsupervised Learning Studio v1.0.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def _detect_identifier_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect potential identifier columns in a dataset.

    Identifier columns typically contain record identifiers and
    are usually not useful for clustering or dimensionality reduction.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        list[str]: List of potential identifier column names.
    """

    identifier_keywords = [
        "id",
        "index",
        "record",
        "unnamed"
    ]

    identifier_columns = []

    for column in df.columns:
        column_lower = column.lower()

        if any(
            keyword in column_lower
            for keyword in identifier_keywords
        ):
            identifier_columns.append(column)

    return identifier_columns   

def _detect_feature_types(
    df: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """
    Detect numeric and categorical feature columns in a dataset.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists:
            - List of numeric feature column names.
            - List of categorical feature column names.
    """

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    return numeric_columns, categorical_columns



def _build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str]
) -> ColumnTransformer:
    """
    Build a ColumnTransformer for preprocessing numeric and categorical features.

    Args:
        numeric_columns (list[str]): List of numeric feature column names.
        categorical_columns (list[str]): List of categorical feature column names.

    Returns:
        ColumnTransformer: A ColumnTransformer that applies appropriate preprocessing
            to numeric and categorical features.
    """


    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
              categorical_columns)
        ]
    )

    return preprocessor

def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Allow the user to select features for analysis.

    Displays available feature columns and prompts the user
    to select which columns should be included in the analysis.

    If the user provides no input, all available features
    are selected by default.

    Args:
        df (pd.DataFrame): Input dataset containing available features.

    Returns:
        pd.DataFrame: Dataset containing only the selected features.
    """

    available_features = df.columns.tolist()

    while True:
        print("\nAvailable Features:")

        for index, feature in enumerate(available_features, start=1):
            print(f"{index}. {feature}")

        user_input = input(
            "\nSelect features (comma-separated, leave blank for all): "
        ).strip()

        if not user_input:
            return df.copy()

        try:
            selected_indices = [
                int(index.strip()) - 1
                for index in user_input.split(",")
            ]

            if not all(
                0 <= index < len(available_features)
                for index in selected_indices
            ):
                print("Invalid selection. Please try again.")
                continue

            selected_features = [
                available_features[index]
                for index in selected_indices
            ]

            return df[selected_features].copy()

        except ValueError:
            print("Invalid input. Please enter feature numbers only.")

def preprocess_data(
    df: pd.DataFrame
) -> tuple[np.ndarray, ColumnTransformer]:
    """
    Preprocess a dataset for unsupervised learning analysis.

    This function orchestrates the complete preprocessing workflow,
    including identifier column detection, feature selection,
    feature type detection, preprocessing pipeline construction,
    and data transformation.

    Workflow:
        1. Detect potential identifier columns.
        2. Optionally exclude identifier columns.
        3. Allow the user to select features.
        4. Detect numeric and categorical features.
        5. Build a preprocessing pipeline.
        6. Fit and transform the selected data.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        tuple[np.ndarray, ColumnTransformer]:
            - Processed feature matrix ready for clustering
              or dimensionality reduction.
            - Fitted preprocessing pipeline.

    Raises:
        ValueError: If preprocessing cannot be completed due
            to invalid feature selection or dataset structure.
    """
   
    identifier_columns = _detect_identifier_columns(df)

    if identifier_columns:
        print("\nDetected Identifier Columns:")
        for column in identifier_columns:
            print(f"- {column}")

        while True:
            exclude_identifiers = input(
                "\nDo you want to exclude identifier columns? (Y/N): "
            ).strip().lower()

            if exclude_identifiers in {"y", "n"}:
                break

            print("Invalid choice. Please enter Y or N.")

        if exclude_identifiers == "y":
            df = df.drop(columns=identifier_columns)
            print("Identifier columns excluded from analysis.")

    selected_df = _select_features(df)

    if selected_df.empty:
        raise ValueError("No features selected for analysis.")

    numeric_columns, categorical_columns = (
        _detect_feature_types(selected_df)
    )

    preprocessor = _build_preprocessor(
        numeric_columns,
        categorical_columns
    )

    processed_data = np.asarray(
        preprocessor.fit_transform(selected_df)
    )

    return processed_data, preprocessor