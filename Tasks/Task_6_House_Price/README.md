# Task 6 — House Price Prediction

## Objective

Predict **real estate sale prices** using **regression** on property features. The workflow demonstrates end-to-end practice: exploratory analysis, preprocessing (especially **square footage**, **bedrooms**, and **location**), **feature scaling** and **selection**, training **Linear Regression** and **Gradient Boosting**, and evaluation with **MAE** and **RMSE**, plus **actual vs predicted** plots and interpretation of **which features drive price**.

## Dataset

- **File:** `house_price_dataset.csv`
- **Source / style:** Aligned with the [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition: numeric size variables and a **Neighborhood** column for **location**.
- **Bundled data:** The repository includes a CSV with columns **`GrLivArea`**, **`BedroomAbvGr`**, **`TotalBsmtSF`**, **`Neighborhood`**, and **`SalePrice`** so the notebook runs offline. You may replace this file with Kaggle `train.csv` after selecting the same columns.

## Models applied

| Model | Role |
|--------|------|
| **Linear Regression** | Strong baseline; interpretable coefficients on scaled features. |
| **Gradient Boosting Regressor** | Nonlinear ensemble; provides **feature importances** for interpretation. |

Both models use the same **sklearn `Pipeline`**: preprocessing → **`SelectKBest`** (univariate linear association with the target) → regressor.

## Key results (example run)

Metrics depend on the random train/test split (`random_state=42`). After executing `house_price_prediction.ipynb` on the bundled dataset, you should see results in this ballpark:

| Model | MAE (USD) | RMSE (USD) |
|--------|-----------|------------|
| Linear Regression | ~14.6k | ~19.0k |
| Gradient Boosting | ~15.2k | ~19.9k |

On this dataset, **both errors are similar**; relationships are **close to linear**, so a simple linear model is **competitive** with gradient boosting. **RMSE > MAE** indicates some larger errors (typical for heavy-tailed price data).

### Which features matter most?

- **Size:** **`GrLivArea`** (above-ground living square footage) and **`TotalBsmtSF`** (basement area) usually rank high — larger homes tend to sell for more.
- **Location:** **Neighborhood** (one-hot encoded) captures **area-level** price levels; in real Ames data this proxies schools, amenities, and desirability.
- **Bedrooms:** **`BedroomAbvGr`** adds signal but is often **correlated with size**; importance may be split across size features.

See the notebook sections **EDA**, **Gradient Boosting importances**, and **Linear Regression |coefficients|** for tables and plots. Importances are **not causal**; they describe **associations** in the training data.

## Files

| File | Description |
|------|-------------|
| `house_price_prediction.ipynb` | Full pipeline: EDA, preprocessing, scaling, selection, models, metrics, plots, written insights |
| `house_price_dataset.csv` | Input data |
| `README.md` | This summary |

## How to run

From the repository root (with dependencies from `requirements.txt`):

```bash
cd Tasks/Task_6_House_Price
jupyter notebook house_price_prediction.ipynb
```

Headless execution (e.g. CI) may require a non-interactive matplotlib backend:

```bash
cd Tasks/Task_6_House_Price
MPLBACKEND=Agg jupyter nbconvert --to notebook --execute house_price_prediction.ipynb
```

## Insights (what we learn)

1. **EDA** shows price skew and strong **size–price** relationships; **neighborhood** boxplots show **location** effects.
2. **Preprocessing** clips extreme **square footage** tails to stabilize models; **location** is encoded for ML.
3. **MAE / RMSE** quantify average and penalized error in **dollars**; comparing two models shows whether nonlinearity helps.
4. **Actual vs predicted** scatterplots show **calibration**; points should cluster near the diagonal if the model fits well.

This project is for **learning**; production systems would add cross-validation, richer features from the full Kaggle schema, and domain review.
