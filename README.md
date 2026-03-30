# Multivariate GRU for Commodity Price Forecasting

A lightweight deep learning pipeline for weekly commodity price forecasting, designed to work with **small datasets** (a few hundred observations) where conventional wisdom says "just use ARIMA."

## The Premise

> "With limited data, use simple models."

This project challenges that assumption. Instead of defaulting to classical statistical methods, the goal was to **stress-test a recurrent neural network** by carefully constraining its architecture and engineering the right inputs — making deep learning viable even when data is scarce. **This small project tackles the prediction of MOP using MAP as predictor**.

## Architecture

The model uses a **Bidirectional GRU** with a deliberately minimal footprint:

| Layer | Units | Details |
|-------|-------|---------|
| GRU | 8 | `leaky_relu`, return sequences |
| Bidirectional GRU | 4 | `tanh`, L2 regularization (0.01) |
| Dropout | — | 20% |
| Dense | 1 | Linear output |

**Total trainable parameters:** ~500 (depending on feature count)

The small architecture is intentional — with limited data, every free parameter is a potential source of overfitting.

## Key Design Decisions

**Epoch control over regularization** — While L2 and dropout are present in the architecture, the most effective lever against overfitting was directly restricting the number of training epochs. Simple, but it worked.

**Derived features over raw inputs** — Instead of feeding only raw prices, the pipeline generates features derived from the series itself: rate of change, first differences, rolling z-scores, and trend flags. This gives the network richer input representations without requiring external domain knowledge.

**Post-processing corrections** — The pipeline applies bias correction and lag adjustment (via cross-correlation) to the raw predictions, significantly improving alignment with actual values.

**Scenario analysis** — Forecasts are presented as probability distributions, with percentile-based scenarios (P10/P50/P90) to support decision-making under uncertainty.

## Pipeline Overview

```
Raw Data → Outlier Treatment (Winsorization)
         → Feature Engineering (RoC, diff, z-score, trend flags)
         → RobustScaler normalization
         → TimeseriesGenerator (window = 4, batch = 48)
         → GRU Training (300 epochs)
         → Bias Correction + Lag Adjustment
         → 12-week Rolling Forecast
         → Probability Distribution (scenario analysis)
```

## Results

The model achieves a **sMAPE of ~1.8%** on the test set after bias and lag corrections, with stable convergence and no significant overfitting — train loss and validation loss converge together across ~300 epochs.

## Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras** — GRU, Bidirectional, EarlyStopping, ReduceLROnPlateau
- **scikit-learn** — RobustScaler/MinMaxScaler, train/test split
- **scipy** — Cross-correlation for lag detection, statistical distributions
- **pandas / numpy** — Data manipulation and feature engineering
- **matplotlib** — Visualization

## Usage

```python
from MultivaGRU import forecast_cfr_prices

forecast_cfr_prices(
    rawMaterial='MOP',
    dadosHistoricos=df,
    horizonte=12,        # 12-week forecast horizon
    ordem_n=True,        # Enable multivariate features
    preditor='MAP'        # Optional: exogenous predictor variable
)
```

## Files

| File | Description |
|------|-------------|
| `MultivaGRU.py` | Main pipeline — training, evaluation, and forecasting |
| `Dados.xlsx` | Sample dataset |

## License

MIT
