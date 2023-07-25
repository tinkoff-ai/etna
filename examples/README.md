# Examples

We have prepared a set of tutorials for an easy introduction:

## Notebooks

### 01. [Get started](https://github.com/tinkoff-ai/etna/tree/master/examples/01-get_started.ipynb) 
- Loading dataset
- Plotting
- Forecasting single time series
  - Simple forecast
  - Prophet 
  - Catboost
- Forecasting multiple time series 
- Pipelin

### 02. [Backtest](https://github.com/tinkoff-ai/etna/tree/master/examples/02-backtest.ipynb)
- What is backtest and how it works
- How to run a validation
- Backtest with fold masks
- Validation visualisation
- Metrics visualisation

### 03. [EDA](https://github.com/tinkoff-ai/etna/tree/master/examples/03-EDA.ipynb) 
- Loading dataset
- Visualization 
  - Plotting time series 
  - Autocorrelation & partial autocorrelation 
  - Cross-correlation 
  - Correlation heatmap 
  - Distribution 
  - Trend 
  - Seasonality
- Outliers
  - Median method
  - Density method
- Change Points
  - Change points plot
  - Interactive change points plot

### 04. [Regressors and exogenous data](https://github.com/tinkoff-ai/etna/tree/master/examples/04-exogenous_data.ipynb)
- What is regressor? 
  - What is additional data?
- Dataset
  - Loading Dataset
  - EDA
- Forecasting with regressors

### 05. [Custom model and transform](https://github.com/tinkoff-ai/etna/tree/master/examples/05-custom_transform_and_model.ipynb)
- What is transform and how it works
- Custom transform 
  - Per-segment custom transform 
  - Multi-segment custom transform
- Custom model 
  - Creating a new model from scratch 
  - Creating a new model using sklearn interface

### 06. [Deep learning models](https://github.com/tinkoff-ai/etna/tree/master/examples/06-NN_examples.ipynb)
- Loading dataset
- Architecture
- Testing models
  - Baseline 
  - DeepAR 
  - RNN 
  - Deep State Model 
  - N-BEATS Model 
  - PatchTS Model

### 07. [Ensembles](https://github.com/tinkoff-ai/etna/tree/master/examples/07-ensembles.ipynb)
- Loading dataset 
- Building pipelines 
- Ensembles 
  - `VotingEnsemble`
  - `StackingEnsamble`
  - Results

### 08. [Outliers](https://github.com/tinkoff-ai/etna/tree/master/examples/08-outliers.ipynb) 
- Loading dataset 
- Point outliers 
  - Median method 
  - Density method 
  - Prediction interval method 
  - Histogram method 
- Interactive visualization 
- Outliers imputation

### 09. [Forecasting strategies](https://github.com/tinkoff-ai/etna/tree/master/examples/09-forecasting_strategies.ipynb)
- Loading dataset 
- Recursive strategy 
- Direct strategy 
  - `Pipeline`
  - `DirectEnsemble`
- Summary

### 10. [Forecast interpretation](https://github.com/tinkoff-ai/etna/tree/master/examples/10-forecast_interpretation.ipynb)
- Loading dataset
- Forecast decomposition 
  - CatBoost 
  - SARIMAX 
  - BATS 
  - In-sample and out-of-sample decomposition 
- Accessing target components 
- Regressors relevance 
  - Feature relevance 
  - Components relevance

### 11. [Clustering](https://github.com/tinkoff-ai/etna/tree/master/examples/11-clustering.ipynb) 
- Generating dataset 
- Distances 
- Clustering 
  - Building Distance Matrix 
  - Building Clustering algorithm 
  - Predicting clusters 
  - Getting centroids 
- Advanced: Custom Distance 
  - Custom Distance implementation 
  - Custom Distance in clustering

### 12. [AutoML](https://github.com/tinkoff-ai/etna/tree/master/examples/12-automl.ipynb)
- Hyperparameters tuning
  - How `Tune` works
  - Example
- General AutoML
  - How `Auto` works
  - Example
- Summary

### 13. [Inference: using saved pipeline on a new data](https://github.com/tinkoff-ai/etna/tree/master/examples/13-inference.ipynb) 
- Preparing data
- Fitting and saving pipeline 
  - Fitting pipeline 
  - Saving pipeline 
  - Method `to_dict`
- Using saved pipeline on a new data 
  - Loading pipeline 
  - Forecast on a new data

### 14. [Hierarchical time series](https://github.com/tinkoff-ai/etna/tree/master/examples/14-hierarchical_pipeline.ipynb)
- Hierarchical time series 
- Preparing dataset 
  - Manually setting hierarchical structure 
  - Hierarchical structure detection 
- Reconciliation methods 
  - Bottom-up approach 
  - Top-down approach 
- Exogenous variables for hierarchical forecasts

### 15. [Classification](https://github.com/tinkoff-ai/etna/tree/master/examples/15-classification.ipynb)
- Classification 
  - Loading dataset
  - Feature extraction 
  - Cross validation 
- Predictability analysis
  - Loading dataset 
  - Loading pretrained analyzer 
  - Analyzing segments predictability

### 16. [Feature selection](https://github.com/tinkoff-ai/etna/tree/master/examples/16-feature_selection.ipynb)
- Loading dataset
- Feature selection methods
  - Intro to feature selection
  - `TreeFeatureSelectionTransform`
  - `GaleShapleyFeatureSelectionTransform`
  - `MRMRFeatureSelectionTransform`
- Summary

## Scripts

### 01. Hyperparameter search
- [Optuna](https://github.com/tinkoff-ai/etna/tree/master/examples/optuna)
- [WandB sweeps](https://github.com/tinkoff-ai/etna/tree/master/examples/wandb/sweeps) example based on [Hydra](https://hydra.cc/)
