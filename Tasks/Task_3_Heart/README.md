# Task 3 — Heart Disease Classification (Logistic Regression + Decision Tree)

## Objective
Build and evaluate classification models to predict **heart disease presence** from clinical features.

## Dataset
- Local file: `heart.csv`
- Target column: `target` (typically \(1=\) disease, \(0=\) no disease)

## Preprocessing
- Train/test split
- Scaling:
  - **Logistic Regression** benefits from feature scaling (StandardScaler)
  - **Decision Tree** does not require scaling, but we keep preprocessing consistent and comparable

## Models used
- **Logistic Regression** (`sklearn.linear_model.LogisticRegression`)
- **Decision Tree Classifier** (`sklearn.tree.DecisionTreeClassifier`)

## Evaluation
For both models, the notebook reports:
- **Confusion Matrix**
- **Classification Report** (Accuracy, Precision, Recall, F1-score)

## Advanced additions (CV, tuning, ROC-AUC)
The notebook also includes:
- **Cross-validation (Stratified K-Fold)** with aggregated metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- **Hyperparameter tuning** using `GridSearchCV` (optimized for ROC-AUC)
  - Tuned **Logistic Regression** (scaled pipeline; `C` and regularization mix via `l1_ratio`)
  - Tuned **Decision Tree** (`max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`)
- **Feature importance**
  - Decision Tree `feature_importances_`
  - Logistic Regression coefficient magnitudes (on standardized features)
- **ROC-AUC + ROC curves** for tuned models on the test set

## How to run
From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

Open: `Tasks/Task_3_Heart/Task_3_Heart.ipynb`
