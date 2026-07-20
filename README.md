# Unsupervised Learning Studio v1.0

A modular machine learning application for clustering, dimensionality reduction, evaluation, and visualization of unlabeled datasets.

This project is part of the ML Studio roadmap and provides a complete workflow for unsupervised learning experiments using KMeans, DBSCAN, and Agglomerative Clustering.

---

## Features

### Dataset Management
- Load CSV datasets
- Dataset validation
- Missing value detection
- Duplicate column detection
- Minimum row and feature validation

### Data Preprocessing
- Identifier column detection
- Feature selection
- Numeric and categorical feature detection
- Standard scaling
- One-hot encoding

### Clustering Algorithms
- KMeans
- DBSCAN
- Agglomerative Clustering

### Clustering Evaluation
- Silhouette Score
- Calinski-Harabasz Score
- Davies-Bouldin Score

### Visualizations
- Cluster Scatter Plot
- PCA Cluster Plot

### Output Management
- Automatic artifact directory creation
- Figure saving with timestamps
- Organized project outputs

---

## Project Structure

```text
unsupervised-learning-studio/
│
├── artifacts/
│   ├── figures/
│   ├── models/
│   └── reports/
│
├── data/
│
├── src/
│   ├── clustering.py
│   ├── config.py
│   ├── constants.py
│   ├── evaluator.py
│   ├── loader.py
│   ├── preprocessing.py
│   ├── utils.py
│   ├── validator.py
│   └── visualizer.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/aakhmadkhujaev/unsupervised-learning-studio.git
cd unsupervised-learning-studio
```

### Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run the application:

```bash
python main.py
```

Provide the path to your dataset when prompted.

Example:

```text
data/Mall_Customers.csv
```

---

## Supported Algorithms

| Algorithm | Description |
|------------|------------|
| KMeans | Partition-based clustering |
| DBSCAN | Density-based clustering |
| Agglomerative Clustering | Hierarchical clustering |

---

## Evaluation Metrics

### Silhouette Score
Measures cluster separation and cohesion.

Higher values indicate better clustering.

### Calinski-Harabasz Score
Measures cluster compactness and separation.

Higher values indicate better clustering.

### Davies-Bouldin Score
Measures similarity between clusters.

Lower values indicate better clustering.

---

## Example Workflow

```text
Load Dataset
      ↓
Validate Dataset
      ↓
Preprocess Data
      ↓
Select Clustering Algorithm
      ↓
Generate Cluster Labels
      ↓
Evaluate Clustering
      ↓
Generate Visualizations
      ↓
Save Results
```

---

## Example Dataset

Recommended dataset for testing:

Mall Customers Dataset

https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib

---
## Results

### Cluster Scatter Plot

![Cluster Scatter Plot](assets/cluster_scatter_plot.png)

### PCA Cluster Plot

![PCA Cluster Plot](assets/pca_cluster_plot.png)
## Future Improvements

### Version 1.1
- Report generation
- Additional clustering metrics
- Improved visualizations

### Version 2.0
- UMAP visualization
- t-SNE visualization
- Interactive dashboards
- Model persistence
- Automated cluster recommendations

---

## Learning Objectives

This project demonstrates:

- Unsupervised learning workflows
- Clustering algorithms
- Dimensionality reduction
- Machine learning preprocessing
- Model evaluation
- Data visualization
- Modular software architecture

---

## Author

Abror Akhmadkhujaev

Software Engineering Student | Machine Learning Developer

Part of the ML Studio project series.