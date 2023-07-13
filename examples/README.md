## Examples

We have prepared a set of tutorials for an easy introduction:

#### 01. [Get started](https://github.com/tinkoff-ai/etna/tree/master/examples/get_started.ipynb) 
- Creating TSDataset and time series plotting 
- Forecast single time series - Simple forecast, Prophet, Catboost
- Forecast multiple time series
- Pipeline

#### 02. [Backtest](https://github.com/tinkoff-ai/etna/tree/master/examples/backtest.ipynb)
- What is backtest and how it works
- How to run a validation
- Validation visualisation

#### 03. [EDA](https://github.com/tinkoff-ai/etna/tree/master/examples/EDA.ipynb) 
- Visualization
  - Plot
  - Partial autocorrelation
  - Cross-correlation
  - Correlation heatmap
  - Distribution
- Outliers
  - Median method
  - Density method
- Change Points
  - Change points plot
  - Interactive change points plot

#### 04. [Regressors and exogenous data](https://github.com/tinkoff-ai/etna/tree/master/examples/exogenous_data.ipynb)
- What is regressor? 
  - What is exogenous data?
- Dataset
  - Loading Dataset
  - EDA
- Forecast with regressors

#### 05. [Custom model and transform](https://github.com/tinkoff-ai/etna/tree/master/examples/custom_transform_and_model.ipynb)
- What is Transform and how it works 
- Custom Transform 
  - Per-segment Custom Transform 
  - Multi-segment Custom Transform 
- Custom Model

#### 06. [Deep learning models](https://github.com/tinkoff-ai/etna/tree/master/examples/NN_examples.ipynb)
- Creating TSDataset  
- Architecture
- Testing models
  - DeepAR 
  - TFT
  - Simple Model

#### 07. [Ensembles](https://github.com/tinkoff-ai/etna/tree/master/examples/ensembles.ipynb)
- VotingEnsemble
- StackingEnsemble

#### 08. [Outliers](https://github.com/tinkoff-ai/etna/tree/master/examples/outliers.ipynb) 
- Point outliers
  - Median method
  - Density method
  - Prediction interval method
  - Histogram method
- Sequence outliers
- Interactive visualization
- Outliers imputation

#### 09. [Forecasting strategies](https://github.com/tinkoff-ai/etna/tree/master/examples/forecasting_strategies.ipynb)
- Imports and constants 
- Load dataset 
- Recursive strategy 
  - AutoRegressivePipeline 
- Direct strategy 
  - Pipeline 
  - DirectEnsemble 
  - assemble_pipelines + DirectEnsemble 
- Summary

#### 10. [Forecast interpretation](https://github.com/tinkoff-ai/etna/tree/master/examples/forecast_interpretation.ipynb)
- Forecast decomposition 
  - CatBoost 
  - SARIMAX 
  - BATS 
  - In-sample and out-of-sample decomposition 
- Accessing target components 
- Regressors relevance 
  - Feature relevance 
  - Components relevance

#### 11. [Clustering](https://github.com/tinkoff-ai/etna/tree/master/examples/clustering.ipynb) 
- Clustering pipeline
- Custom Distance
- Visualisation

#### 12. [AutoML script](https://github.com/tinkoff-ai/etna/tree/master/examples/auto.py)
- Auto pipeline search

#### 13. [AutoML notebook](https://github.com/tinkoff-ai/etna/tree/master/examples/automl.ipynb)
- Hyperparameters tuning
  - How `Tune` works
  - Example
- General AutoML
  - How `Auto` works
  - Example

#### 14. Hyperparameter search
- [Optuna](https://github.com/tinkoff-ai/etna/tree/master/examples/optuna)
- [WandB sweeps](https://github.com/tinkoff-ai/etna/tree/master/examples/wandb/sweeps) example based on [Hydra](https://hydra.cc/)

#### 15. [Inference: using saved pipeline on a new data](https://github.com/tinkoff-ai/etna/tree/master/examples/inference.ipynb) 
- Fitting and saving pipeline
- Using saved pipeline on a new data

#### 16. [Hierarchical time series](https://github.com/tinkoff-ai/etna/tree/master/examples/hierarchical_pipeline.ipynb)
- Hierarchical time series
- Hierarchical structure
- Reconciliation methods
- Exogenous variables for hierarchical forecasts

#### 17. [Classification](https://github.com/tinkoff-ai/etna/tree/master/examples/classification.ipynb)
- Classification 
  - Load Dataset
  - Feature extraction 
  - Cross validation 
- Predictability analysis
  - Load Dataset 
  - Load pretrained analyzer 
  - Analyze segments predictability

#### 18. [Feature selection](https://github.com/tinkoff-ai/etna/tree/master/examples/feature_selection.ipynb)
- Loading Dataset
- Feature selection methods
  - Intro to feature selection
  - `TreeFeatureSelectionTransform`
  - `GaleShapleyFeatureSelectionTransform`
  - `MRMRFeatureSelectionTransform`
