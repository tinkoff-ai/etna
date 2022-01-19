# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Added
- Method TSDataset.info ([#409](https://github.com/tinkoff-ai/etna/pull/409))
- DifferencingTransform ([#414](https://github.com/tinkoff-ai/etna/pull/414))
- OneHotEncoderTransform and LabelEncoderTransform ([#431](https://github.com/tinkoff-ai/etna/pull/431))
- MADTransform ([#441](https://github.com/tinkoff-ai/etna/pull/441))
- `MRMRFeatureSelectionTransform` ([#439](https://github.com/tinkoff-ai/etna/pull/439))
- Possibility to change metric representation in backtest using Metric.name ([#454](https://github.com/tinkoff-ai/etna/pull/454))

## Changed
- Change method TSDataset.describe ([#409](https://github.com/tinkoff-ai/etna/pull/409))
- Group Transforms according to their impact ([#420](https://github.com/tinkoff-ai/etna/pull/420))
- Change the way `LagTransform`, `DateFlagsTransform` and `TimeFlagsTransform` generate column names ([#421](https://github.com/tinkoff-ai/etna/pull/421))
- Clarify the behaviour of TimeSeriesImputerTransform in case of all NaN values ([#427](https://github.com/tinkoff-ai/etna/pull/427))
- Fixed bug in title in `sample_acf_plot` method ([#432](https://github.com/tinkoff-ai/etna/pull/432))
- Pytorch-forecasting and sklearn version update + some pytroch transform API changing ([#445](https://github.com/tinkoff-ai/etna/pull/445))

### Fixed
- Add relevance_params in GaleShapleyFeatureSelectionTransform ([#410](https://github.com/tinkoff-ai/etna/pull/410))
- Docs for statistics transforms ([#441](https://github.com/tinkoff-ai/etna/pull/441))
- SARIMAX parameters fix ([#459](https://github.com/tinkoff-ai/etna/pull/459))

## [1.5.0] - 2021-12-24
### Added
- Holiday Transform ([#359](https://github.com/tinkoff-ai/etna/pull/359))
- S3FileLogger and LocalFileLogger ([#372](https://github.com/tinkoff-ai/etna/pull/372))
- Parameter `changepoint_prior_scale` to `ProphetModel` ([#408](https://github.com/tinkoff-ai/etna/pull/408))

### Changed
- Set `strict_optional = True` for mypy ([#381](https://github.com/tinkoff-ai/etna/pull/381))
- Move checking the series endings to `make_future` step ([#413](https://github.com/tinkoff-ai/etna/pull/413)) 

### Fixed
- Sarimax bug in future prediction with quantiles ([#391](https://github.com/tinkoff-ai/etna/pull/391))
- Catboost version too high ([#394](https://github.com/tinkoff-ai/etna/pull/394))
- Add sorting of classes in left bar in docs ([#397](https://github.com/tinkoff-ai/etna/pull/397))
- nn notebook in docs ([#396](https://github.com/tinkoff-ai/etna/pull/396))
- SklearnTransform column name generation ([#398](https://github.com/tinkoff-ai/etna/pull/398))
- Inverse transform doesn't affect quantiles ([#395](https://github.com/tinkoff-ai/etna/pull/395))

## [1.4.2] - 2021-12-09
### Fixed
- Docs generation for neural networks

## [1.4.1] - 2021-12-09
### Changed
- Speed up `_check_regressors` and `_merge_exog` ([#360](https://github.com/tinkoff-ai/etna/pull/360))

### Fixed
- `Model`, `PerSegmentModel`, `PerSegmentWrapper` imports ([#362](https://github.com/tinkoff-ai/etna/pull/362))
- Docs generation ([#363](https://github.com/tinkoff-ai/etna/pull/363))
- Fixed work of get_anomalies_density with constant series ([#334](https://github.com/tinkoff-ai/etna/issues/334))

## [1.4.0] - 2021-12-03
### Added
- ACF plot ([#318](https://github.com/tinkoff-ai/etna/pull/318))

### Changed
- Add `ts.inverse_transform` as final step at `Pipeline.fit` method ([#316](https://github.com/tinkoff-ai/etna/pull/316))
- Make test_ts optional in plot_forecast ([#321](https://github.com/tinkoff-ai/etna/pull/321))
- Speed up inference for multisegment regression models ([#333](https://github.com/tinkoff-ai/etna/pull/333))
- Speed up Pipeline._get_backtest_forecasts ([#336](https://github.com/tinkoff-ai/etna/pull/336))
- Speed up SegmentEncoderTransform ([#331](https://github.com/tinkoff-ai/etna/pull/331))
- Wandb Logger does not work unless pytorch is installed ([#340](https://github.com/tinkoff-ai/etna/pull/340))

### Fixed
- Get rid of lambda in DensityOutliersTransform and get_anomalies_density ([#341](https://github.com/tinkoff-ai/etna/pull/341))
- Fixed import in transforms ([#349](https://github.com/tinkoff-ai/etna/pull/349))
- Pickle DTWClustering ([#350](https://github.com/tinkoff-ai/etna/pull/350))

### Removed
- Remove TimeSeriesCrossValidation ([#337](https://github.com/tinkoff-ai/etna/pull/337))

## [1.3.3] - 2021-11-24
### Added
- RelevanceTable returns rank ([#268](https://github.com/tinkoff-ai/etna/pull/268/))
- GaleShapleyFeatureSelectionTransform ([#284](https://github.com/tinkoff-ai/etna/pull/284))
- FilterFeaturesTransform ([#277](https://github.com/tinkoff-ai/etna/pull/277))
- Spell checking for source code and md files ([#303](https://github.com/tinkoff-ai/etna/pull/303))
- ResampleWithDistributionTransform ([#296](https://github.com/tinkoff-ai/etna/pull/296))
- Add function to duplicate exogenous data ([#305](https://github.com/tinkoff-ai/etna/pull/305))
- FourierTransform ([#306](https://github.com/tinkoff-ai/etna/pull/306))

### Changed
- Rename confidence interval to prediction interval, start working with quantiles instead of interval_width ([#285](https://github.com/tinkoff-ai/etna/pull/285))
- Changed format of forecast and test dataframes in WandbLogger ([#309](https://github.com/tinkoff-ai/etna/pull/309))

### Fixed

## [1.3.2] - 2021-11-18
### Changed
- Add sum for omegaconf resolvers ([#300](https://github.com/tinkoff-ai/etna/pull/300/))

## [1.3.1] - 2021-11-12
### Changed
- Delete restriction on version of pandas ([#274](https://github.com/tinkoff-ai/etna/pull/274))

## [1.3.0] - 2021-11-12
### Added
- Backtest cli ([#223](https://github.com/tinkoff-ai/etna/pull/223), [#259](https://github.com/tinkoff-ai/etna/pull/259))
- TreeFeatureSelectionTransform ([#229](https://github.com/tinkoff-ai/etna/pull/229))
- Feature relevance table calculation using tsfresh ([#227](https://github.com/tinkoff-ai/etna/pull/227), [#249](https://github.com/tinkoff-ai/etna/pull/249))
- Method to_flatten to TSDataset ([#241](https://github.com/tinkoff-ai/etna/pull/241)
- Out_column parameter to not inplace transforms([#211](https://github.com/tinkoff-ai/etna/pull/211))
- omegaconf config parser in cli ([#258](https://github.com/tinkoff-ai/etna/pull/258))
- Feature relevance table calculation using feature importance ([#261](https://github.com/tinkoff-ai/etna/pull/261))
- MeanSegmentEncoderTransform ([#265](https://github.com/tinkoff-ai/etna/pull/265))

### Changed
- Add possibility to set custom in_column for ConfidenceIntervalOutliersTransform ([#240](https://github.com/tinkoff-ai/etna/pull/240))
- Make `in_column` the first argument in every transform ([#247](https://github.com/tinkoff-ai/etna/pull/247))
- Update mypy checking and fix issues with it ([#248](https://github.com/tinkoff-ai/etna/pull/248))
- Add histogram method in outliers notebook ([#252](https://github.com/tinkoff-ai/etna/pull/252)) 
- Joblib parameters for backtest and ensembles ([#253](https://github.com/tinkoff-ai/etna/pull/253))
- Replace cycle over segments with vectorized expression in TSDataset._check_endings ([#264](https://github.com/tinkoff-ai/etna/pull/264))

### Fixed
- Fixed broken links in docs command section ([#223](https://github.com/tinkoff-ai/etna/pull/223))
- Fix default value for TSDataset.tail ([#245](https://github.com/tinkoff-ai/etna/pull/245))
- Fix raising warning on fitting SklearnModel on dataset categorical columns ([#250](https://github.com/tinkoff-ai/etna/issues/207)) 
- Fix working TSDataset.make_future with empty exog values ([#244](https://github.com/tinkoff-ai/etna/pull/244))
- Fix issue with aggregate_metrics=True for ConsoleLogger and WandbLogger ([#254](https://github.com/tinkoff-ai/etna/pull/254))
- Fix binder requirements to work with optional dependencies ([#257](https://github.com/tinkoff-ai/etna/pull/257))

## [1.2.0] - 2021-10-27
### Added
- BinsegTrendTransform, ChangePointsTrendTransform ([#87](https://github.com/tinkoff-ai/etna/pull/87))
- Interactive plot for anomalies (#[95](https://github.com/tinkoff-ai/etna/pull/95))
- Examples to TSDataset methods with doctest ([#92](https://github.com/tinkoff-ai/etna/pull/92))
- WandbLogger ([#71](https://github.com/tinkoff-ai/etna/pull/71))
- Pipeline ([#78](https://github.com/tinkoff-ai/etna/pull/78))
- Sequence anomalies ([#96](https://github.com/tinkoff-ai/etna/pull/96)), Histogram anomalies ([#79](https://github.com/tinkoff-ai/etna/pull/79))
- 'is_weekend' feature in DateFlagsTransform ([#101](https://github.com/tinkoff-ai/etna/pull/101))
- Documentation example for models and note about inplace nature of forecast ([#112](https://github.com/tinkoff-ai/etna/pull/112))
- Property regressors to TSDataset ([#82](https://github.com/tinkoff-ai/etna/pull/82))
- Clustering ([#110](https://github.com/tinkoff-ai/etna/pull/110))
- Outliers notebook ([#123](https://github.com/tinkoff-ai/etna/pull/123)))
- Method inverse_transform in TimeSeriesImputerTransform ([#135](https://github.com/tinkoff-ai/etna/pull/135))
- VotingEnsemble ([#150](https://github.com/tinkoff-ai/etna/pull/150))
- Forecast command for cli ([#133](https://github.com/tinkoff-ai/etna/issues/133))
- MyPy checks in CI/CD and lint commands ([#39](https://github.com/tinkoff-ai/etna/issues/39))
- TrendTransform ([#139](https://github.com/tinkoff-ai/etna/pull/139))
- Running notebooks in ci ([#134](https://github.com/tinkoff-ai/etna/issues/134))
- Cluster plotter to EDA ([#169](https://github.com/tinkoff-ai/etna/pull/169))
- Pipeline.backtest method ([#161](https://github.com/tinkoff-ai/etna/pull/161), [#192](https://github.com/tinkoff-ai/etna/pull/192))
- STLTransform class ([#158](https://github.com/tinkoff-ai/etna/pull/158))
- NN_examples notebook ([#159](https://github.com/tinkoff-ai/etna/pull/159))
- Example for ProphetModel ([#178](https://github.com/tinkoff-ai/etna/pull/178))
- Instruction notebook for custom model and transform creation ([#180](https://github.com/tinkoff-ai/etna/pull/180))
- Add inverse_transform in *OutliersTransform ([#160](https://github.com/tinkoff-ai/etna/pull/160))
- Examples for CatBoostModelMultiSegment and CatBoostModelPerSegment ([#181](https://github.com/tinkoff-ai/etna/pull/181))
- Simplify TSDataset.train_test_split method by allowing to pass not all values ([#191](https://github.com/tinkoff-ai/etna/pull/191))
- Confidence interval anomalies detection to EDA ([#182](https://github.com/tinkoff-ai/etna/pull/182))
- ConfidenceIntervalOutliersTransform ([#196](https://github.com/tinkoff-ai/etna/pull/196))
- Add 'in_column' parameter to get_anomalies methods([#199](https://github.com/tinkoff-ai/etna/pull/199))
- Clustering notebook ([#152](https://github.com/tinkoff-ai/etna/pull/152))
- StackingEnsemble ([#195](https://github.com/tinkoff-ai/etna/pull/195))
- Add AutoRegressivePipeline ([#209](https://github.com/tinkoff-ai/etna/pull/209))
- Ensembles notebook ([#218](https://github.com/tinkoff-ai/etna/pull/218))
- Function plot_backtest_interactive ([#225](https://github.com/tinkoff-ai/etna/pull/225))
- Confidence intervals in Pipeline ([#221](https://github.com/tinkoff-ai/etna/pull/221)) 

### Changed
- Delete offset from WindowStatisticsTransform ([#111](https://github.com/tinkoff-ai/etna/pull/111))
- Add Pipeline example in Get started notebook ([#115](https://github.com/tinkoff-ai/etna/pull/115))
- Internal implementation of BinsegTrendTransform ([#141](https://github.com/tinkoff-ai/etna/pull/141))
- Colorebar scaling in Correlation heatmap plotter ([#143](https://github.com/tinkoff-ai/etna/pull/143))
- Add Correlation heatmap in EDA notebook ([#144](https://github.com/tinkoff-ai/etna/pull/144))
- Add `__repr__` for Pipeline ([#151](https://github.com/tinkoff-ai/etna/pull/151))
- Defined random state for every test cases ([#155](https://github.com/tinkoff-ai/etna/pull/155))
- Add confidence intervals to Prophet ([#153](https://github.com/tinkoff-ai/etna/pull/153))
- Add confidence intervals to SARIMA ([#172](https://github.com/tinkoff-ai/etna/pull/172))
- Add badges to all example notebooks ([#220](https://github.com/tinkoff-ai/etna/pull/220)) 
- Update backtest notebook by adding Pipeline.backtest ([222](https://github.com/tinkoff-ai/etna/pull/222))

### Fixed
- Set default value of `TSDataset.head` method ([#170](https://github.com/tinkoff-ai/etna/pull/170))
- Categorical and fillna issues with pandas >=1.2 ([#190](https://github.com/tinkoff-ai/etna/pull/190))
- Fix `TSDataset.to_dataset` method sorting bug ([#183](https://github.com/tinkoff-ai/etna/pull/183))
- Undefined behaviour of DataFrame.loc[:, pd.IndexSlice[:, ["a", "b"]]] between 1.1.* and >= 1.2 ([#188](https://github.com/tinkoff-ai/etna/pull/188))
- Fix typo in word "length" in `get_segment_sequence_anomalies`,`get_sequence_anomalies`,`SAXOutliersTransform` arguments ([#212](https://github.com/tinkoff-ai/etna/pull/212))
- Make possible to send backtest plots with many segments ([#225](https://github.com/tinkoff-ai/etna/pull/225))

## [1.1.3] - 2021-10-08
### Fixed
- Limit version of pandas by 1.2 (excluding) ([#163](https://github.com/tinkoff-ai/etna/pull/163))

## [1.1.2] - 2021-10-08
### Changed
- SklearnTransform out column names ([#99](https://github.com/tinkoff-ai/etna/pull/99))
- Update EDA notebook ([#96](https://github.com/tinkoff-ai/etna/pull/96))
- Add 'regressor_' prefix to output columns of LagTransform, DateFlagsTransform, SpecialDaysTransform, SegmentEncoderTransform
### Fixed
- Add more obvious Exception Error for forecasting with unfitted model ([#102](https://github.com/tinkoff-ai/etna/pull/102))
- Fix bug with hardcoded frequency in PytorchForecastingTransform ([#107](https://github.com/tinkoff-ai/etna/pull/107))
- Bug with inverse_transform method of TimeSeriesImputerTransform ([#148](https://github.com/tinkoff-ai/etna/pull/148))

## [1.1.1] - 2021-09-23
### Fixed
- Documentation build workflow ([#85](https://github.com/tinkoff-ai/etna/pull/85))

## [1.1.0] - 2021-09-23
### Added 
- MedianOutliersTransform, DensityOutliersTransform ([#30](https://github.com/tinkoff-ai/etna/pull/30))
- Issues and Pull Request templates
- TSDataset checks ([#24](https://github.com/tinkoff-ai/etna/pull/24), [#20](https://github.com/tinkoff-ai/etna/pull/20))\
- Pytorch-Forecasting models ([#29](https://github.com/tinkoff-ai/etna/pull/29))
- SARIMAX model ([#10](https://github.com/tinkoff-ai/etna/pull/10))
- Logging, including ConsoleLogger ([#46](https://github.com/tinkoff-ai/etna/pull/46))
- Correlation heatmap plotter ([#77](https://github.com/tinkoff-ai/etna/pull/77))

### Changed
- Backtest is fully parallel 
- New default hyperparameters for CatBoost
- Add 'regressor_' prefix to output columns of LagTransform, DateFlagsTransform, SpecialDaysTransform, SegmentEncoderTransform

### Fixed
- Documentation fixes ([#55](https://github.com/tinkoff-ai/etna/pull/55), [#53](https://github.com/tinkoff-ai/etna/pull/53), [#52](https://github.com/tinkoff-ai/etna/pull/52))
- Solved warning in LogTransform and AddConstantTransform ([#26](https://github.com/tinkoff-ai/etna/pull/26))
- Regressors do not have enough history bug ([#35](https://github.com/tinkoff-ai/etna/pull/35))
- make_future(1) and make_future(2) bug
- Fix working with 'cap' and 'floor' features in Prophet model ([#62](https://github.com/tinkoff-ai/etna/pull/62))
- Fix saving init params for SARIMAXModel ([#81](https://github.com/tinkoff-ai/etna/pull/81))
- Imports of nn models, PytorchForecastingTransform and Transform ([#80](https://github.com/tinkoff-ai/etna/pull/80))

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
  - Sklearn's scalers (MinMax, Robust, MinMaxAbs, Standard, MaxAbs)
  - BoxCox, YeoJohnson, LogTransform
  - Lag operator
  - NaN imputer
- TimeSeriesCrossValidation
- Time Series Dataset (TSDataset)
- Playground datasets generation (AR, constant, periodic, from pattern)
- Metrics (MAE, MAPE, SMAPE, MedAE, MSE, MSLE, R^2)
- EDA methods
  - Outliers detection
  - PACF plot
  - Cross correlation plot
  - Distribution plot
  - Anomalies (Outliers) plot
  - Backtest (CrossValidation) plot
  - Forecast plot
