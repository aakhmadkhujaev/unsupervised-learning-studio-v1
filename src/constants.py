
"""
Constants used throughout Unsupervised Learning Studio v1.0.
"""

# ==========================================================
# Main Menu
# ==========================================================

MAIN_MENU_OPTIONS = {
    "1": "Clustering Analysis",
    "2": "Dimensionality Reduction Analysis",
    "3": "Complete Analysis",
    "4": "Exit"
}

# ==========================================================
# Supported Algorithms
# ==========================================================

SUPPORTED_CLUSTERING_MODELS = {
    "1": {
        "name": "KMeans"
    },
    "2": {
        "name": "DBSCAN"
    },
    "3": {
        "name": "Agglomerative Clustering"
    }
}

SUPPORTED_REDUCTION_MODELS = {
    "1": {
        "name": "PCA"
    },
    "2": {
        "name": "t-SNE"
    }
}

# ==========================================================
# Validation
# ==========================================================

MINIMUM_ROWS = 10
MINIMUM_FEATURES = 2

# ==========================================================
# Dimensionality Reduction
# ==========================================================

DEFAULT_PCA_COMPONENTS = 2
DEFAULT_TSNE_COMPONENTS = 2

# ==========================================================
# Reports
# ==========================================================

CLUSTER_REPORT_NAME = "cluster_report.txt"
REDUCTION_REPORT_NAME = "reduction_report.txt"