
"""
Constants used throughout Unsupervised Learning Studio v1.0.
"""

# ==========================================================
# Main Menu
# ==========================================================

from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans


from sklearn.cluster import AgglomerativeClustering

from config import RANDOM_STATE


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
        "name": "KMeans",
        "model": KMeans,
        "params": {
            "n_clusters": 3,
            "random_state": RANDOM_STATE
        }
    },
    "2": {
        "name": "DBSCAN",
        "model": DBSCAN,
        "params": {
            "eps": 0.5,
            "min_samples": 5
        }
    },
    "3": {
        "name": "Agglomerative Clustering",
        "model": AgglomerativeClustering,
        "params": {
            "n_clusters": 3
        }
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