# Task 1 — Iris Dataset (EDA + Visualization)

## Objective
Perform **Exploratory Data Analysis (EDA)** on the Iris dataset and create clear, portfolio-ready visualizations to understand feature distributions, class separability, and potential outliers.

## Dataset
- **Primary**: Local `Iris.csv` / `iris.csv` (the notebook searches common locations).
- **Fallback**: `seaborn.load_dataset("iris")` if a local CSV is not found.

### Expected columns
The local CSV commonly contains:
- `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`, `Species` (plus an optional `Id`)

The seaborn dataset uses:
- `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `species`

The notebook normalizes column names so the analysis is consistent.

## What’s inside the notebook
### EDA
- `.head()`
- `.info()`
- `.describe()`
- Missing-value check and basic sanity validation

### Visualizations (Seaborn/Matplotlib)
- **Scatter plot**: sepal length vs sepal width (colored by species)
- **Histograms**: feature distributions (with optional KDE)
- **Box plots**: outlier inspection per feature across species
- **Correlation heatmap**: feature correlation matrix (Pearson)
- **Pairplot**: pairwise feature plots colored by species

## Models used
This task is primarily **EDA-focused**, but the notebook also includes **baseline classifiers** to complement the analysis:
- **Logistic Regression** (with `StandardScaler`)
- **K-Nearest Neighbors (KNN)** (with `StandardScaler`)
- **Support Vector Machine (SVM, RBF kernel)** (with `StandardScaler`)

### Evaluation (for baselines)
- Accuracy comparison across models
- Classification Report (Precision/Recall/F1)
- Confusion Matrix for the best-performing model

## How to run
From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name devhub-internship --display-name "Python (devhub-internship)"
jupyter notebook
```

Open: `Tasks/Task_1_Iris/Task_1_Iris.ipynb`
