# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Backtest cli ([#223](https://github.com/tinkoff-ai/etna-ts/pull/223))
- TreeFeatureSelectionTransform ([#229](https://github.com/tinkoff-ai/etna-ts/pull/229))
- Feature relevance table calculation ([#227](https://github.com/tinkoff-ai/etna-ts/pull/227), [#249](https://github.com/tinkoff-ai/etna-ts/pull/249))
- Method flatten to TSDataset ([#241](https://github.com/tinkoff-ai/etna-ts/pull/241)

### Changed
- Add possibility to set custom in_column for ConfidenceIntervalOutliersTransform ([#240](https://github.com/tinkoff-ai/etna-ts/pull/240))
- Make `in_column` the first argument in every transform ([#247](https://github.com/tinkoff-ai/etna-ts/pull/247))
- Update mypy checking and fix issues with it ([#248](https://github.com/tinkoff-ai/etna-ts/pull/248))

### Fixed
- Fixed broken links in docs command section ([#223](https://github.com/tinkoff-ai/etna-ts/pull/223))
- Fix default value for TSDataset.tail ([#245](https://github.com/tinkoff-ai/etna-ts/pull/245))
- Fix working TSDataset.make_future with empty exog values ([#244](https://github.com/tinkoff-ai/etna-ts/pull/244))

## [1.2.0] - 2021-10-27
### Added
- BinsegTrendTransform, ChangePointsTrendTransform ([#87](https://github.com/tinkoff-ai/etna-ts/pull/87))
- Interactive plot for anomalies (#[95](https://github.com/tinkoff-ai/etna-ts/pull/95))
- Examples to TSDataset methods with doctest ([#92](https://github.com/tinkoff-ai/etna-ts/pull/92))
- WandbLogger ([#71](https://github.com/tinkoff-ai/etna-ts/pull/71))
- Pipeline ([#78](https://github.com/tinkoff-ai/etna-ts/pull/78))
- Sequence anomalies ([#96](https://github.com/tinkoff-ai/etna-ts/pull/96)), Histogram anomalies ([#79](https://github.com/tinkoff-ai/etna-ts/pull/79))
- 'is_weekend' feature in DateFlagsTransform ([#101](https://github.com/tinkoff-ai/etna-ts/pull/101))
- Documentation example for models and note about inplace nature of forecast ([#112](https://github.com/tinkoff-ai/etna-ts/pull/112))
- Property regressors to TSDataset ([#82](https://github.com/tinkoff-ai/etna-ts/pull/82))
- Clustering ([#110](https://github.com/tinkoff-ai/etna-ts/pull/110))
- Outliers notebook ([#123](https://github.com/tinkoff-ai/etna-ts/pull/123)))
- Method inverse_transform in TimeSeriesImputerTransform ([#135](https://github.com/tinkoff-ai/etna-ts/pull/135))
- VotingEnsemble ([#150](https://github.com/tinkoff-ai/etna-ts/pull/150))
- Forecast command for cli ([#133](https://github.com/tinkoff-ai/etna-ts/issues/133))
- MyPy checks in CI/CD and lint commands ([#39](https://github.com/tinkoff-ai/etna-ts/issues/39))
- TrendTransform ([#139](https://github.com/tinkoff-ai/etna-ts/pull/139))
- Running notebooks in ci ([#134](https://github.com/tinkoff-ai/etna-ts/issues/134))
- Cluster plotter to EDA ([#169](https://github.com/tinkoff-ai/etna-ts/pull/169))
- Pipeline.backtest method ([#161](https://github.com/tinkoff-ai/etna-ts/pull/161), [#192](https://github.com/tinkoff-ai/etna-ts/pull/192))
- STLTransform class ([#158](https://github.com/tinkoff-ai/etna-ts/pull/158))
- NN_examples notebook ([#159](https://github.com/tinkoff-ai/etna-ts/pull/159))
- Example for ProphetModel ([#178](https://github.com/tinkoff-ai/etna-ts/pull/178))
- Instruction notebook for custom model and transform creation ([#180](https://github.com/tinkoff-ai/etna-ts/pull/180))
- Add inverse_transform in *OutliersTransform ([#160](https://github.com/tinkoff-ai/etna-ts/pull/160))
- Examples for CatBoostModelMultiSegment and CatBoostModelPerSegment ([#181](https://github.com/tinkoff-ai/etna-ts/pull/181))
- Simplify TSDataset.train_test_split method by allowing to pass not all values ([#191](https://github.com/tinkoff-ai/etna-ts/pull/191))
- Confidence interval anomalies detection to EDA ([#182](https://github.com/tinkoff-ai/etna-ts/pull/182))
- ConfidenceIntervalOutliersTransform ([#196](https://github.com/tinkoff-ai/etna-ts/pull/196))
- Add 'in_column' parameter to get_anomalies methods([#199](https://github.com/tinkoff-ai/etna-ts/pull/199))
- Clustering notebook ([#152](https://github.com/tinkoff-ai/etna-ts/pull/152))
- StackingEnsemble ([#195](https://github.com/tinkoff-ai/etna-ts/pull/195))
- Add AutoRegressivePipeline ([#209](https://github.com/tinkoff-ai/etna-ts/pull/209))
- Ensembles notebook ([#218](https://github.com/tinkoff-ai/etna-ts/pull/218))
- Function plot_backtest_interactive ([#225](https://github.com/tinkoff-ai/etna-ts/pull/225))
- Confidence intervals in Pipeline ([#221](https://github.com/tinkoff-ai/etna-ts/pull/221)) 

### Changed
- Delete offset from WindowStatisticsTransform ([#111](https://github.com/tinkoff-ai/etna-ts/pull/111))
- Add Pipeline example in Get started notebook ([#115](https://github.com/tinkoff-ai/etna-ts/pull/115))
- Internal implementation of BinsegTrendTransform ([#141](https://github.com/tinkoff-ai/etna-ts/pull/141))
- Colorebar scaling in Correlation heatmap plotter ([#143](https://github.com/tinkoff-ai/etna-ts/pull/143))
- Add Correlation heatmap in EDA notebook ([#144](https://github.com/tinkoff-ai/etna-ts/pull/144))
- Add `__repr__` for Pipeline ([#151](https://github.com/tinkoff-ai/etna-ts/pull/151))
- Defined random state for every test cases ([#155](https://github.com/tinkoff-ai/etna-ts/pull/155))
- Add confidence intervals to Prophet ([#153](https://github.com/tinkoff-ai/etna-ts/pull/153))
- Add confidence intervals to SARIMA ([#172](https://github.com/tinkoff-ai/etna-ts/pull/172))
- Add badges to all example notebooks ([#220](https://github.com/tinkoff-ai/etna-ts/pull/220)) 
- Update backtest notebook by adding Pipeline.backtest ([222](https://github.com/tinkoff-ai/etna-ts/pull/222))

### Fixed
- Set default value of `TSDataset.head` method ([#170](https://github.com/tinkoff-ai/etna-ts/pull/170))
- Categorical and fillna issues with pandas >=1.2 ([#190](https://github.com/tinkoff-ai/etna-ts/pull/190))
- Fix `TSDataset.to_dataset` method sorting bug ([#183](https://github.com/tinkoff-ai/etna-ts/pull/183))
- Undefinded behaviour of DataFrame.loc[:, pd.IndexSlice[:, ["a", "b"]]] between 1.1.* and >= 1.2 ([#188](https://github.com/tinkoff-ai/etna-ts/pull/188))
- Fix typo in word "length" in `get_segment_sequence_anomalies`,`get_sequence_anomalies`,`SAXOutliersTransform` arguments ([#212](https://github.com/tinkoff-ai/etna-ts/pull/212))
- Make possible to send backtest plots with many segments ([#225](https://github.com/tinkoff-ai/etna-ts/pull/225))

## [1.1.3] - 2021-10-08
### Fixed
- Limit version of pandas by 1.2 (excluding) ([#163](https://github.com/tinkoff-ai/etna-ts/pull/163))

## [1.1.2] - 2021-10-08
### Changed
- SklearnTransform out column names ([#99](https://github.com/tinkoff-ai/etna-ts/pull/99))
- Update EDA notebook ([#96](https://github.com/tinkoff-ai/etna-ts/pull/96))
- Add 'regressor_' prefix to output columns of LagTransform, DateFlagsTransform, SpecialDaysTransform, SegmentEncoderTransform
### Fixed
- Add more obvious Exception Error for forecasting with unfitted model ([#102](https://github.com/tinkoff-ai/etna-ts/pull/102))
- Fix bug with hardcoded frequency in PytorchForecastingTransform ([#107](https://github.com/tinkoff-ai/etna-ts/pull/107))
- Bug with inverse_transform method of TimeSeriesImputerTransform ([#148](https://github.com/tinkoff-ai/etna-ts/pull/148))

## [1.1.1] - 2021-09-23
### Fixed
- Documentation build workflow ([#85](https://github.com/tinkoff-ai/etna-ts/pull/85))

## [1.1.0] - 2021-09-23
### Added 
- MedianOutliersTransform, DensityOutliersTransform ([#30](https://github.com/tinkoff-ai/etna-ts/pull/30))
- Issues and Pull Request templates
- TSDataset checks ([#24](https://github.com/tinkoff-ai/etna-ts/pull/24), [#20](https://github.com/tinkoff-ai/etna-ts/pull/20))\
- Pytorch-Forecasting models ([#29](https://github.com/tinkoff-ai/etna-ts/pull/29))
- SARIMAX model ([#10](https://github.com/tinkoff-ai/etna-ts/pull/10))
- Logging, including ConsoleLogger ([#46](https://github.com/tinkoff-ai/etna-ts/pull/46))
- Correlation heatmap plotter ([#77](https://github.com/tinkoff-ai/etna-ts/pull/77))

### Changed
- Backtest is fully parallel 
- New default hyperparameters for CatBoost
- Add 'regressor_' prefix to output columns of LagTransform, DateFlagsTransform, SpecialDaysTransform, SegmentEncoderTransform

### Fixed
- Documentation fixes ([#55](https://github.com/tinkoff-ai/etna-ts/pull/55), [#53](https://github.com/tinkoff-ai/etna-ts/pull/53), [#52](https://github.com/tinkoff-ai/etna-ts/pull/52))
- Solved warning in LogTransform and AddConstantTransform ([#26](https://github.com/tinkoff-ai/etna-ts/pull/26))
- Regressors does not have enough history bug ([#35](https://github.com/tinkoff-ai/etna-ts/pull/35))
- make_future(1) and make_future(2) bug
- Fix working with 'cap' and 'floor' features in Prophet model ([#62](https://github.com/tinkoff-ai/etna-ts/pull/62)))
- Fix saving init params for SARIMAXModel ([#81](https://github.com/tinkoff-ai/etna-ts/pull/81))
- Imports of nn models, PytorchForecastingTransform and Transform ([#80](https://github.com/tinkoff-ai/etna-ts/pull/80)))

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
