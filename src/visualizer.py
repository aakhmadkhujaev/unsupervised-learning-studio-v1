"""
Visualization utilities for Unsupervised Learning Studio v1.0.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from src.constants import DEFAULT_PCA_COMPONENTS
from config import (
    FIGURE_SIZE,
    FIGURE_DPI
)

from src.utils import save_figure


def plot_clusters(
    data: np.ndarray,
    labels: np.ndarray
) -> Path:
    """
    Create and save a cluster scatter plot.

    This visualization displays cluster assignments in a
    two-dimensional feature space. If the dataset contains
    more than two features, only the first two dimensions
    are used for plotting.

    Args:
        data (np.ndarray): Feature matrix used for clustering.
        labels (np.ndarray): Cluster labels assigned to samples.

    Returns:
        Path: Path to the saved visualization file.
    """
    if data.shape[1] < 2:
        raise ValueError("Dataset must have at least two features for scatter plot visualization.")
    plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title('Cluster Scatter Plot')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter, label="Cluster")
    figure_path = save_figure('cluster_scatter_plot')
    return figure_path


def plot_pca_clusters(
    data: np.ndarray,
    labels: np.ndarray
) -> Path:
    """
    Create and save a PCA-based cluster visualization.

    This visualization reduces the dataset to two principal
    components and displays cluster assignments in the
    resulting two-dimensional space.

    Args:
        data (np.ndarray): Feature matrix used for clustering.
        labels (np.ndarray): Cluster labels assigned to samples.

    Returns:
        Path: Path to the saved visualization file.
    """
    pca = PCA(n_components=DEFAULT_PCA_COMPONENTS)
    reduced_data = pca.fit_transform(data)

    plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title('PCA Cluster Plot')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label="Cluster")
    figure_path = save_figure('pca_cluster_plot')
    return figure_path

def generate_visualizations(
    data: np.ndarray,
    labels: np.ndarray
) -> dict[str, Path]:
    """
    Generate clustering visualizations.

    This function orchestrates the visualization workflow,
    including cluster scatter plots and PCA-based cluster
    visualizations.

    Workflow:
        1. Generate cluster scatter plot.
        2. Generate PCA cluster plot.
        3. Return saved figure paths.

    Args:
        data (np.ndarray): Feature matrix used for clustering.
        labels (np.ndarray): Cluster labels assigned to samples.

    Returns:
        dict[str, Path]: Paths to generated visualization files.
    """
    visualizations = {}
    visualizations['cluster_scatter_plot'] = plot_clusters(data, labels)
    visualizations['pca_cluster_plot'] = plot_pca_clusters(data, labels)
    return visualizations