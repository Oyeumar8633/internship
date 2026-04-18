# Task 2 — Stock Price Prediction (AAPL, Linear Regression)

## Objective
Download **historical stock data for Apple (AAPL)** and build a simple baseline model to predict the **next-day closing price** using **Linear Regression**.

## Dataset
- Source: `yfinance` (Yahoo Finance)
- Ticker: `AAPL`
- Time range: **last 2 years**
- Frequency: daily bars

## Problem framing
This is a **supervised regression** setup:
- **Features (X)**: same-day market data (e.g., Open/High/Low/Close/Volume, optionally returns/moving averages)
- **Target (y)**: `Target = Close(t + 1)` (next trading day close)

The notebook uses a **time-based split** (no shuffling) to reduce leakage.

## Model used
- **Linear Regression** (`sklearn.linear_model.LinearRegression`)

This is intentionally a **baseline** model. The goal is to demonstrate a correct ML workflow (data collection → preprocessing → modeling → evaluation → visualization).

## Evaluation & visualization
- Metrics (typical): MAE / RMSE (and/or \(R^2\))
- Plot: **Actual vs Predicted** closing prices on the test range

## Implementation details (important)
- The notebook **flattens `yfinance` columns** when they come back as a MultiIndex / tuple columns (common across versions), so `Open/High/Low/Close/Volume` work reliably.
- RMSE is computed in a **version-safe** way as \(\sqrt{\text{MSE}}\) for compatibility with older `scikit-learn` versions.

## How to run
From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

Open: `Tasks/Task_2_Stock/Task_2_Stock.ipynb`

## Notes
- `yfinance` requires an internet connection.
- Market data can contain missing days (weekends/holidays). The notebook handles this naturally via the downloaded index.
