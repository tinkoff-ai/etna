# Examples

We have prepared a set of tutorials for an easy introduction:

[Quickstart](https://github.com/tinkoff-ai/etna/tree/master/examples/quick_start.ipynb)

## Tutorials

### Basic

#### [Get started](https://github.com/tinkoff-ai/etna/tree/master/examples/101-get_started.ipynb) 
- Loading dataset
- Plotting
- Forecasting single time series
  - Naive forecast
  - Prophet 
  - Catboost
- Forecasting multiple time series 

#### [Backtest](https://github.com/tinkoff-ai/etna/tree/master/examples/102-backtest.ipynb)
- What is backtest and how it works
- How to run a validation
- Backtest with fold masks
- Validation visualisation
- Metrics visualisation

#### [EDA](https://github.com/tinkoff-ai/etna/tree/master/examples/103-EDA.ipynb) 
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

## Intermediate

#### [Regressors and exogenous data](https://github.com/tinkoff-ai/etna/tree/master/examples/201-exogenous_data.ipynb)
- What is regressor? 
  - What is additional data?
- Dataset
  - Loading Dataset
  - EDA
- Forecasting with regressors

#### [Deep learning models](https://github.com/tinkoff-ai/etna/tree/master/examples/202-NN_examples.ipynb)
- Loading dataset
- Architecture
- Testing models
  - Baseline 
  - DeepAR 
  - RNN 
  - Deep State Model 
  - N-BEATS Model 
  - PatchTS Model

#### [Ensembles](https://github.com/tinkoff-ai/etna/tree/master/examples/203-ensembles.ipynb)
- Loading dataset 
- Building pipelines 
- Ensembles 
  - `VotingEnsemble`
  - `StackingEnsamble`
  - Results

#### [Outliers](https://github.com/tinkoff-ai/etna/tree/master/examples/204-outliers.ipynb) 
- Loading dataset 
- Point outliers 
  - Median method 
  - Density method 
  - Prediction interval method 
  - Histogram method 
- Interactive visualization 
- Outliers imputation

#### [AutoML](https://github.com/tinkoff-ai/etna/tree/master/examples/205-automl.ipynb)
- Hyperparameters tuning
  - How `Tune` works
  - Example
- General AutoML
  - How `Auto` works
  - Example
- Summary

#### [Clustering](https://github.com/tinkoff-ai/etna/tree/master/examples/206-clustering.ipynb) 
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

#### [Feature selection](https://github.com/tinkoff-ai/etna/tree/master/examples/207-feature_selection.ipynb)
- Loading dataset
- Feature selection methods
  - Intro to feature selection
  - `TreeFeatureSelectionTransform`
  - `GaleShapleyFeatureSelectionTransform`
  - `MRMRFeatureSelectionTransform`
- Summary

#### [Forecasting strategies](https://github.com/tinkoff-ai/etna/tree/master/examples/208-forecasting_strategies.ipynb)
- Loading dataset 
- Recursive strategy 
- Direct strategy 
  - `Pipeline`
  - `DirectEnsemble`
- Summary


#### [Mechanics of forecasting](https://github.com/tinkoff-ai/etna/tree/master/examples/209-mechanics_of_forecasting.ipynb)
- Loading dataset
- Forecasting
  - Context-free models
  - Context-required models
  - ML models
- Summary

### Advanced

#### [Custom model and transform](https://github.com/tinkoff-ai/etna/tree/master/examples/301-custom_transform_and_model.ipynb)
- What is transform and how it works
- Custom transform 
  - Per-segment custom transform 
  - Multi-segment custom transform
- Custom model 
  - Creating a new model from scratch 
  - Creating a new model using sklearn interface

#### [Inference: using saved pipeline on a new data](https://github.com/tinkoff-ai/etna/tree/master/examples/302-inference.ipynb) 
- Preparing data
- Fitting and saving pipeline 
  - Fitting pipeline 
  - Saving pipeline 
  - Method `to_dict`
- Using saved pipeline on a new data 
  - Loading pipeline 
  - Forecast on a new data

#### [Hierarchical time series](https://github.com/tinkoff-ai/etna/tree/master/examples/303-hierarchical_pipeline.ipynb)
- Hierarchical time series 
- Preparing dataset 
  - Manually setting hierarchical structure 
  - Hierarchical structure detection 
- Reconciliation methods 
  - Bottom-up approach 
  - Top-down approach 
- Exogenous variables for hierarchical forecasts

#### [Forecast interpretation](https://github.com/tinkoff-ai/etna/tree/master/examples/304-forecast_interpretation.ipynb)
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

#### [Classification](https://github.com/tinkoff-ai/etna/tree/master/examples/305-classification.ipynb)
- Classification 
  - Loading dataset
  - Feature extraction 
  - Cross validation 
- Predictability analysis
  - Loading dataset 
  - Loading pretrained analyzer 
  - Analyzing segments predictability

## Scripts

### Hyperparameter search
- [Optuna](https://github.com/tinkoff-ai/etna/tree/master/examples/optuna)
- [WandB sweeps](https://github.com/tinkoff-ai/etna/tree/master/examples/wandb/sweeps) example based on [Hydra](https://hydra.cc/)
