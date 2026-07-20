from src.constants import SUPPORTED_CLUSTERING_MODELS

import numpy as np

from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering
)
from typing import Any
from config import RANDOM_STATE

def _get_clustering_model(
    algorithm_choice: str
) -> object:
    """
    Create and return a clustering model based on the
    selected algorithm.

    Supported algorithms:
        1. KMeans
        2. DBSCAN
        3. Agglomerative Clustering

    Args:
        algorithm_choice (str): Selected clustering algorithm.

    Returns:
        object: Initialized clustering model.

    Raises:
        ValueError: If an unsupported algorithm is selected.
    """
    
    if algorithm_choice not in SUPPORTED_CLUSTERING_MODELS:
        raise ValueError(f"Unsupported clustering algorithm: {algorithm_choice}")

    model_config = SUPPORTED_CLUSTERING_MODELS[algorithm_choice]

    model_class = model_config["model"]

    params = model_config["params"]

    return model_class(**params)

def _fit_clustering_model(
    model: Any,
    data: np.ndarray
) -> np.ndarray:
    """
    Fit a clustering model and generate cluster labels.

    Args:
        model (Any): Initialized clustering model.
        data (np.ndarray): Preprocessed feature matrix.

    Returns:
        np.ndarray: Cluster labels assigned to each sample.
    """
    labels = model.fit_predict(data)
    return labels

def run_clustering(
    data: np.ndarray,
    algorithm_choice: str
) -> tuple[Any, np.ndarray]:
    """
    Run a clustering analysis on preprocessed data.

    This function orchestrates the clustering workflow,
    including model selection, model fitting, and
    cluster label generation.

    Workflow:
        1. Select clustering algorithm.
        2. Initialize clustering model.
        3. Fit model to data.
        4. Generate cluster labels.

    Args:
        data (np.ndarray): Preprocessed feature matrix.
        algorithm_choice (str): Selected clustering algorithm.

    Returns:
        tuple[Any, np.ndarray]:
            - Trained clustering model.
            - Cluster labels assigned to each sample.

    Raises:
        ValueError: If an unsupported clustering algorithm
            is selected.
    """
    model = _get_clustering_model(algorithm_choice)
    labels = _fit_clustering_model(model, data)
    return model, labels