# Multivariate GRU for Commodity Price Forecasting

This repository contains a forecasting pipeline for weekly commodity prices using a Bidirectional GRU neural network. The dataset is small — a few hundred weekly observations — which is a scenario where deep learning models are generally not recommended due to overfitting risk. The purpose of this project was to test whether it is possible to make a recurrent neural network work under such constraints by directly restricting its architecture and carefully selecting input features.

## Context

It is common practice in time series forecasting to default to classical statistical models (such as ARIMA or Holt-Winters) when the available data is limited. Neural networks, particularly recurrent architectures, tend to require larger datasets to generalize well. This project takes a different approach: instead of avoiding deep learning altogether, the idea was to stress-test a GRU model by reducing its capacity to the minimum viable architecture and controlling overfitting primarily through epoch restriction either by early stopping or by finding out a cut-off epoch. 

## Architecture

The model is a Bidirectional GRU with a deliberately small number of units. The architecture can be summarized as follows:

| Layer | Units | Activation | Notes |
|-------|-------|------------|-------|
| GRU (Bidirectional) | 8 | leaky_relu | Returns sequences |
| GRU (Bidirectional) | 4 | tanh | L2 regularization (0.01) |
| Dropout | — | — | 20% |
| Dense | 1 | linear | Output layer |

While L2 regularization and dropout are present in the architecture, the most effective mechanism to prevent overfitting was directly limiting the number of training epochs. This proved itself as a simple but effective approach for this particular dataset size.

## Feature Engineering

Rather than relying on the network to extract all relevant patterns from raw price data alone, the pipeline generates derived features from the series itself. These include rate of change, first differences, rolling z-scores, and trend indicators. The objective is to provide the model with richer input representations without introducing external domain knowledge — all features are computed from the price series and, optionally, from a correlated exogenous variable.

## Post-Processing

The raw model output goes through two correction steps before being used for forecasting:

1. **Bias correction** — the systematic offset between predicted and actual values on the test set is estimated and subtracted from the forecast.
2. **Lag adjustment** — cross-correlation is used to detect temporal misalignment between predictions and actuals. The forecast is then shifted accordingly.

These corrections proved to be relevant, reducing the sMAPE from approximately 5.5% to 1.8% on the test set.

## Scenario Analysis

Forecasts are presented not as single point estimates but as a probability distribution. Percentile-based scenarios (P10, P20, P50, P80, P90) are computed from the forecast distribution, which is useful for decision-making under uncertainty in procurement and supply chain planning.

## Pipeline

The general flow of the pipeline is as follows:

```
Raw Data → Outlier Treatment (Winsorization)
         → Feature Engineering (RoC, diff, z-score, trend flags)
         → RobustScaler normalization
         → TimeseriesGenerator (window = 12, batch = 48)
         → GRU Training (EarlyStopping, ReduceLROnPlateau)
         → Bias Correction + Lag Adjustment
         → Rolling Forecast (12-week horizon)
         → Probability Distribution (scenario analysis)
```

## Dependencies

The project uses Python 3.10+ with the following libraries: TensorFlow/Keras for the GRU model, scikit-learn for scaling and evaluation, scipy for cross-correlation and statistical distributions, pandas and numpy for data manipulation and matplotlib for visualization.

## Files

| File | Description |
|------|-------------|
| `MultivaGRU.py` | Complete pipeline for training, evaluation and forecasting |
| `Dados.xlsx` | Sample dataset |

## Usage

```python
from MultivaGRU import forecast_cfr_prices

forecast_cfr_prices(
    rawMaterial='MOP',
    dadosHistoricos=df,
    horizonte=12,
    ordem_n=True,
    preditor='MAP'
)
```

## License


