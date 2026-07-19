"""
Clustering evaluation utilities for Unsupervised Learning Studio v1.0.
"""

import numpy as np

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


def _count_clusters(
    labels: np.ndarray
) -> int:
    """
    Count the number of actual clusters in a clustering result.

    Noise points labeled as -1 are excluded from the count.

    Args:
        labels (np.ndarray): Cluster labels assigned to samples.

    Returns:
        int: Number of detected clusters excluding noise.
    """
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)  # Exclude noise label
    return len(unique_labels)


def _validate_clustering_result(
    labels: np.ndarray
) -> None:
    """
    Validate that a clustering result contains enough clusters
    for evaluation metrics.

    Args:
        labels (np.ndarray): Cluster labels assigned to samples.

    Raises:
        ValueError: If fewer than two clusters are detected.
    """
    num_clusters = _count_clusters(labels)
    if num_clusters < 2:
        raise ValueError(
            f"Clustering result must contain at least two clusters for evaluation. "
            f"Detected clusters: {num_clusters}."
        )


def _calculate_clustering_metrics(
    data: np.ndarray,
    labels: np.ndarray
) -> dict[str, float]:
    """
    Calculate clustering evaluation metrics.

    Args:
        data (np.ndarray): Feature matrix used for clustering.
        labels (np.ndarray): Cluster labels assigned to samples.

    Returns:
        dict[str, float]: Dictionary containing clustering metrics.
    """
    metrics = {
        "silhouette_score": silhouette_score(data, labels),
        "calinski_harabasz_score": calinski_harabasz_score(data, labels),
        "davies_bouldin_score": davies_bouldin_score(data, labels)
    }
    return metrics

def evaluate_clustering(
    data: np.ndarray,
    labels: np.ndarray
) -> dict[str, float]:
    """
    Evaluate clustering performance using internal validation metrics.

    This function orchestrates the clustering evaluation workflow,
    including cluster validation and metric calculation.

    Workflow:
        1. Validate clustering result.
        2. Calculate evaluation metrics.
        3. Return evaluation results.

    Args:
        data (np.ndarray): Feature matrix used for clustering.
        labels (np.ndarray): Cluster labels assigned to samples.

    Returns:
        dict[str, float]: Clustering evaluation metrics.

    Raises:
        ValueError: If the clustering result is not valid
            for evaluation.
    """
    _validate_clustering_result(labels)

    metrics = _calculate_clustering_metrics(
        data,
        labels
     )

    return metrics