
from pathlib import Path

from src.utils import (
    print_header,
    create_output_directories,
    print_section
)

from src.loader import load_dataset

from src.validator import validate_dataset

from src.preprocessing import preprocess_data

from src.clustering import run_clustering

from src.evaluator import evaluate_clustering

from src.visualizer import generate_visualizations

from src.constants import (
    SUPPORTED_CLUSTERING_MODELS
)
def _display_main_menu() -> str:
    """
    Display the clustering algorithm menu and
    return the user's selection.

    Returns:
        str: Selected clustering algorithm key.
    """
    while True:
        print_header("Clustering Algorithm Selection")

        for key, model_info in (
            SUPPORTED_CLUSTERING_MODELS.items()
        ):
            print(f"{key}. {model_info['name']}")

        choice = input(
            "\nEnter your choice: "
        ).strip()

        if choice in SUPPORTED_CLUSTERING_MODELS:
            return choice

        print("Invalid choice. Please try again.")
def _run_clustering_workflow(
    file_path: str
) -> None:
    """
    Execute the complete clustering workflow.

    This function orchestrates dataset loading,
    validation, preprocessing, clustering,
    evaluation, and visualization generation.

    Workflow:
        1. Load dataset.
        2. Validate dataset.
        3. Preprocess dataset.
        4. Select clustering algorithm.
        5. Run clustering analysis.
        6. Evaluate clustering results.
        7. Generate visualizations.
        8. Display results.

    Args:
        file_path (str): Path to the input dataset.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If dataset validation or clustering fails.
    """
    df = load_dataset(file_path)

    validate_dataset(df)

    processed_data, preprocessor = preprocess_data(df)

    algorithm_choice = _display_main_menu()

    model, labels = run_clustering(
        processed_data,
        algorithm_choice
    )

    metrics = evaluate_clustering(
        processed_data,
        labels
    )

    visualizations = generate_visualizations(
        processed_data,
        labels
    )
    print_header("Clustering Results")
    algorithm_name = (
        SUPPORTED_CLUSTERING_MODELS[
            algorithm_choice
        ]["name"]
    )
    cluster_count = len(
        set(labels) - {-1}
    )
    print_section("Clustering Summary")
    print(f"Algorithm: {algorithm_name}")
    print(f"Number of Clusters: {cluster_count}")
    print_section("Evaluation Metrics")

    for metric, value in metrics.items():
        metric_name = metric.replace("_", " ").title()

        print(
            f"{metric_name}: {value:.4f}"
        )
    print_section("Generated Visualizations")

    for name, path in visualizations.items():
        visualization_name = (
            name.replace("_", " ").title()
        )

        print(
            f"{visualization_name}: {path}"
        )

def _get_dataset_path() -> str:
    """
    Prompt the user for a dataset path.

    Returns:
        str: Valid dataset file path.
    """
    while True:
        file_path = input(
            "Enter the path to your dataset (CSV format): "
        ).strip()

        if Path(file_path).exists() and Path(file_path).is_file():
            return file_path
    
        print("Invalid input. Please provide a valid file path.")
def main() -> None:
    """
    Run the Unsupervised Learning Studio application.
    """
    print_header("Welcome to Unsupervised Learning Studio")

    create_output_directories()

    dataset_path = _get_dataset_path()

    _run_clustering_workflow(dataset_path)
if __name__ == "__main__":
    main()