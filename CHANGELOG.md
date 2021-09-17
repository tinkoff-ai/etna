# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added 

-

### Changed

-

### Fixed

-

### Removed

-

## [1.0.0] - 2021-09-05
### Added
- Models
  - CatBoost
  - Prophet
  - Seasonal Moving Average
  - Naive
  - Linear
- Transforms
  - Rolling statistics
  - Trend removal
  - Segment encoder
  - Datetime flags
  - Sklearn skalers (MinMax, Robust, MinMaxAbs, Standard, MaxAbs)
  - BoxCox, YeoJohnson, LogTransform
  - Lag operator
  - NaN imputer
- TimeSeriesCrossValidation
- Time Series Dataset (TSDataset)
- Playground datasets generation (AR, constant, periodic, from pattern)
- Matrics (MAE, MAPE, SMAPE, MedAE, MSE, MSLE, R^2)
- EDA mehods
  - Outliers detection
  - PACF plot
  - Cross correlation plot
  - Destribution plot
  - Anomalies (Outliers) plot
  - Backtest (CrossValidation) plot
  - Forecast plot
