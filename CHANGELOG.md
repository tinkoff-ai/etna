# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Added
-
-
-
-

### Changed
-
-
-
-

### Fixed
-
-
-
-

### Removed
-

## [2.2.0] - 2023-08-08
### Added
- `DeseasonalityTransform` ([#1307](https://github.com/tinkoff-ai/etna/pull/1307))
- Add extension with models from `statsforecast`: `StatsForecastARIMAModel`, `StatsForecastAutoARIMAModel`, `StatsForecastAutoCESModel`, `StatsForecastAutoETSModel`, `StatsForecastAutoThetaModel` ([#1295](https://github.com/tinkoff-ai/etna/pull/1297))
- Notebook `feature_selection` ([#875](https://github.com/tinkoff-ai/etna/pull/875))
- Implementation of PatchTS model ([#1277](https://github.com/tinkoff-ai/etna/pull/1277))

### Changed
- Add modes `binary` and `category` to `HolidayTransform` ([#763](https://github.com/tinkoff-ai/etna/pull/763))
- Add sorting by timestamp before the fit in `CatBoostPerSegmentModel` and `CatBoostMultiSegmentModel` ([#1337](https://github.com/tinkoff-ai/etna/pull/1337))
- Speed up metrics computation by optimizing segment validation, forbid NaNs during metrics computation ([#1338](https://github.com/tinkoff-ai/etna/pull/1338))
- Unify errors, warnings and checks in models ([#1312](https://github.com/tinkoff-ai/etna/pull/1312))
- Remove upper limitation on version of numba ([#1321](https://github.com/tinkoff-ai/etna/pull/1321))
- Optimize `TSDataset.describe` and `TSDataset.info` by vectorization ([#1344](https://github.com/tinkoff-ai/etna/pull/1344))
- Add documentation warning about using dill during loading ([#1346](https://github.com/tinkoff-ai/etna/pull/1346))
- Vectorize metric computation ([#1347](https://github.com/tinkoff-ai/etna/pull/1347))

### Fixed
- Pipeline ensembles fail in `etna forecast` CLI ([#1331](https://github.com/tinkoff-ai/etna/pull/1331))
- Fix performance of `DeepARModel` and `TFTModel` ([#1322](https://github.com/tinkoff-ai/etna/pull/1322))
- `mrmr` feature selection working with categoricals ([#1311](https://github.com/tinkoff-ai/etna/pull/1311))
- Fix version of `statsforecast` to 1.4 to avoid dependency conflicts during installation ([#1313](https://github.com/tinkoff-ai/etna/pull/1313))
- Add inverse transformation into `predict` method of pipelines ([#1314](https://github.com/tinkoff-ai/etna/pull/1314))
- Allow saving large pipelines ([#1335](https://github.com/tinkoff-ai/etna/pull/1335))
- Fix link for dataset in classification notebook ([#1351](https://github.com/tinkoff-ai/etna/pull/1351))

### Removed
- Building docker images with cuda 10.2 ([#1306](https://github.com/tinkoff-ai/etna/pull/1306))

## [2.1.0] - 2023-06-30
### Added
- Notebook `forecast_interpretation.ipynb` with forecast decomposition ([#1220](https://github.com/tinkoff-ai/etna/pull/1220))
- Exogenous variables shift transform `ExogShiftTransform`([#1254](https://github.com/tinkoff-ai/etna/pull/1254))
- Parameter `start_timestamp` to forecast CLI command ([#1265](https://github.com/tinkoff-ai/etna/pull/1265))
- `DeepStateModel` ([#1253](https://github.com/tinkoff-ai/etna/pull/1253))
- `NBeatsGenericModel` and `NBeatsInterpretableModel` ([#1302](https://github.com/tinkoff-ai/etna/pull/1302))
- Function `estimate_max_n_folds` for folds number estimation ([#1279](https://github.com/tinkoff-ai/etna/pull/1279))
- Parameters `estimate_n_folds` and `context_size` to forecast and backtest CLI commands ([#1284](https://github.com/tinkoff-ai/etna/pull/1284))
- Class `Tune` for hyperparameter optimization within existing pipeline ([#1200](https://github.com/tinkoff-ai/etna/pull/1200))
- Add `etna.distributions` for using it instead of using `optuna.distributions` ([#1292](https://github.com/tinkoff-ai/etna/pull/1292))

### Changed
- Set the default value of `final_model` to `LinearRegression(positive=True)` in the constructor of `StackingEnsemble` ([#1238](https://github.com/tinkoff-ai/etna/pull/1238))
- Add microseconds to `FileLogger`'s directory name ([#1264](https://github.com/tinkoff-ai/etna/pull/1264))
- Inherit `SaveMixin` from `AbstractSaveable` for mypy checker ([#1261](https://github.com/tinkoff-ai/etna/pull/1261))
- Update requirements for `holidays` and `scipy`, change saving library from `pickle` to `dill` in `SaveMixin` ([#1268](https://github.com/tinkoff-ai/etna/pull/1268))
- Update requirement for `ruptures`, add requirement for `sqlalchemy` ([#1276](https://github.com/tinkoff-ai/etna/pull/1276))
- Optimize `make_samples` of `RNNNet` and `MLPNet` ([#1281](https://github.com/tinkoff-ai/etna/pull/1281))
- Remove `to_be_fixed` from inference tests on `SpecialDaysTransform` ([#1283](https://github.com/tinkoff-ai/etna/pull/1283))
- Rewrite `TimeSeriesImputerTransform` to work without per-segment wrapper ([#1293](https://github.com/tinkoff-ai/etna/pull/1293))
- Add default `params_to_tune` for catboost models ([#1185](https://github.com/tinkoff-ai/etna/pull/1185))
- Add default `params_to_tune` for `ProphetModel` ([#1203](https://github.com/tinkoff-ai/etna/pull/1203))
- Add default `params_to_tune` for `SARIMAXModel`, change default parameters for the model ([#1206](https://github.com/tinkoff-ai/etna/pull/1206))
- Add default `params_to_tune` for linear models ([#1204](https://github.com/tinkoff-ai/etna/pull/1204))
- Add default `params_to_tune` for `SeasonalMovingAverageModel`, `MovingAverageModel`, `NaiveModel` and `DeadlineMovingAverageModel` ([#1208](https://github.com/tinkoff-ai/etna/pull/1208))
- Add default `params_to_tune` for `DeepARModel` and `TFTModel` ([#1210](https://github.com/tinkoff-ai/etna/pull/1210))
- Add default `params_to_tune` for `HoltWintersModel`, `HoltModel` and `SimpleExpSmoothingModel` ([#1209](https://github.com/tinkoff-ai/etna/pull/1209))
- Add default `params_to_tune` for `RNNModel` and `MLPModel` ([#1218](https://github.com/tinkoff-ai/etna/pull/1218))
- Add default `params_to_tune` for `DateFlagsTransform`, `TimeFlagsTransform`, `SpecialDaysTransform` and `FourierTransform` ([#1228](https://github.com/tinkoff-ai/etna/pull/1228))
- Add default `params_to_tune` for `MedianOutliersTransform`, `DensityOutliersTransform` and `PredictionIntervalOutliersTransform` ([#1231](https://github.com/tinkoff-ai/etna/pull/1231))
- Add default `params_to_tune` for `TimeSeriesImputerTransform` ([#1232](https://github.com/tinkoff-ai/etna/pull/1232))
- Add default `params_to_tune` for `DifferencingTransform`, `MedianTransform`, `MaxTransform`, `MinTransform`, `QuantileTransform`, `StdTransform`, `MeanTransform`, `MADTransform`, `MinMaxDifferenceTransform`, `SumTransform`, `BoxCoxTransform`, `YeoJohnsonTransform`, `MaxAbsScalerTransform`, `MinMaxScalerTransform`, `RobustScalerTransform` and `StandardScalerTransform` ([#1233](https://github.com/tinkoff-ai/etna/pull/1233))
- Add default `params_to_tune` for `LabelEncoderTransform` ([#1242](https://github.com/tinkoff-ai/etna/pull/1242))
- Add default `params_to_tune` for `ChangePointsSegmentationTransform`, `ChangePointsTrendTransform`, `ChangePointsLevelTransform`, `TrendTransform`, `LinearTrendTransform`, `TheilSenTrendTransform` and `STLTransform` ([#1243](https://github.com/tinkoff-ai/etna/pull/1243))
- Add default `params_to_tune` for `TreeFeatureSelectionTransform`, `MRMRFeatureSelectionTransform` and `GaleShapleyFeatureSelectionTransform` ([#1250](https://github.com/tinkoff-ai/etna/pull/1250))
- Add tuning stage into `Auto.fit` ([#1272](https://github.com/tinkoff-ai/etna/pull/1272))
- Add `params_to_tune` into `Tune` init ([#1282](https://github.com/tinkoff-ai/etna/pull/1282))
- Skip duplicates during `Tune.fit`, skip duplicates in `top_k`, add AutoML notebook ([#1285](https://github.com/tinkoff-ai/etna/pull/1285))
- Add parameter `fast_redundancy` in `mrmm`, fix relevance calculation in `get_model_relevance_table` ([#1294](https://github.com/tinkoff-ai/etna/pull/1294))

### Fixed
- Fix `plot_backtest` and `plot_backtest_interactive` on one-step forecast ([1260](https://github.com/tinkoff-ai/etna/pull/1260))
- Fix `BaseReconciliator` to work on `pandas==1.1.5` ([#1229](https://github.com/tinkoff-ai/etna/pull/1229))
- Fix `TSDataset.make_future` to handle hierarchy, quantiles, target components ([#1248](https://github.com/tinkoff-ai/etna/pull/1248))
- Fix warning during creation of `ResampleWithDistributionTransform` ([#1230](https://github.com/tinkoff-ai/etna/pull/1230))
- Add deep copy for copying attributes of `TSDataset` ([#1241](https://github.com/tinkoff-ai/etna/pull/1241))
- Add `tsfresh` into optional dependencies, remove instruction about `pip install tsfresh` ([#1246](https://github.com/tinkoff-ai/etna/pull/1246))
- Fix `DeepARModel` and `TFTModel` to work with changed `prediction_size` ([#1251](https://github.com/tinkoff-ai/etna/pull/1251))
- Fix problems with flake8 B023 ([#1252](https://github.com/tinkoff-ai/etna/pull/1252))
- Fix problem with swapped forecast methods in HierarchicalPipeline ([#1259](https://github.com/tinkoff-ai/etna/pull/1259))
- Fix problem with segment name "target" in `StackingEnsemble` ([#1262](https://github.com/tinkoff-ai/etna/pull/1262))
- Fix `BasePipeline.forecast` when prediction intervals are estimated on history data with presence of NaNs ([#1291](https://github.com/tinkoff-ai/etna/pull/1291))
- Teach `BaseMixin.set_params` to work with nested `list` and `tuple` ([#1201](https://github.com/tinkoff-ai/etna/pull/1201))
- Fix `get_anomalies_prediction_interval` to work when segments have different start date ([#1296](https://github.com/tinkoff-ai/etna/pull/1296))
- Fix `classification` notebook to download `FordA` dataset without error ([#1299](https://github.com/tinkoff-ai/etna/pull/1299))
- Fix signature of `Auto.fit`, `Tune.fit` to not have a breaking change ([#1300](https://github.com/tinkoff-ai/etna/pull/1300))

## [2.0.0] - 2023-04-11
### Added
- Target components logic into `AutoRegressivePipeline` ([#1188](https://github.com/tinkoff-ai/etna/pull/1188))
- Target components logic into `HierarchicalPipeline` ([#1199](https://github.com/tinkoff-ai/etna/pull/1199))
- `predict` method into `HierarchicalPipeline` ([#1199](https://github.com/tinkoff-ai/etna/pull/1199))
- Add target components handling in `get_level_dataframe` ([#1179](https://github.com/tinkoff-ai/etna/pull/1179))
- Forecast decomposition for `SeasonalMovingAverageModel`([#1180](https://github.com/tinkoff-ai/etna/pull/1180))
- Target components logic into base classes of pipelines ([#1173](https://github.com/tinkoff-ai/etna/pull/1173))
- Method `predict_components` for forecast decomposition in `_SklearnAdapter` and `_LinearAdapter` for linear models ([#1164](https://github.com/tinkoff-ai/etna/pull/1164))
- Target components logic into base classes of models ([#1158](https://github.com/tinkoff-ai/etna/pull/1158))
- Target components logic to TSDataset ([#1153](https://github.com/tinkoff-ai/etna/pull/1153))
- Methods `save` and `load` to HierarchicalPipeline ([#1096](https://github.com/tinkoff-ai/etna/pull/1096))
- New data access methods in `TSDataset` : `update_columns_from_pandas`, `add_columns_from_pandas`, `drop_features` ([#809](https://github.com/tinkoff-ai/etna/pull/809))
- `PytorchForecastingDatasetBuiler` for neural networks from Pytorch Forecasting ([#971](https://github.com/tinkoff-ai/etna/pull/971))
- New base classes for per-segment and multi-segment transforms `IrreversiblePersegmentWrapper`, `ReversiblePersegmentWrapper`, `IrreversibleTransform`, `ReversibleTransform` ([#835](https://github.com/tinkoff-ai/etna/pull/835))
- New base class for one segment transforms `OneSegmentTransform` ([#894](https://github.com/tinkoff-ai/etna/pull/894))
- `ChangePointsLevelTransform` and base classes `PerIntervalModel`, `BaseChangePointsModelAdapter` for per-interval transforms ([#998](https://github.com/tinkoff-ai/etna/pull/998))
- Method `set_params` to change parameters of ETNA objects ([#1102](https://github.com/tinkoff-ai/etna/pull/1102))
- Function `plot_forecast_decomposition` ([#1129](https://github.com/tinkoff-ai/etna/pull/1129))
- Method `forecast_components` for forecast decomposition in `_TBATSAdapter` ([#1133](https://github.com/tinkoff-ai/etna/pull/1133))
- Methods `forecast_components` and `predict_components` for forecast decomposition in `_CatBoostAdapter` ([#1148](https://github.com/tinkoff-ai/etna/pull/1148))
- Methods `forecast_components` and `predict_components` for forecast decomposition in `_HoltWintersAdapter ` ([#1162](https://github.com/tinkoff-ai/etna/pull/1162))
- Method `predict_components` for forecast decomposition in `_ProphetAdapter` ([#1172](https://github.com/tinkoff-ai/etna/pull/1172))
- Methods `forecast_components` and `predict_components` for forecast decomposition in `_SARIMAXAdapter` and `_AutoARIMAAdapter` ([#1174](https://github.com/tinkoff-ai/etna/pull/1174))
- Add `refit` parameter into `backtest` ([#1159](https://github.com/tinkoff-ai/etna/pull/1159))
- Add `stride` parameter into `backtest` ([#1165](https://github.com/tinkoff-ai/etna/pull/1165))
- Add optional parameter `ts` into `forecast` method of pipelines ([#1071](https://github.com/tinkoff-ai/etna/pull/1071))
- Add tests on `transform` method of transforms on subset of segments, on new segments, on future with gap ([#1094](https://github.com/tinkoff-ai/etna/pull/1094))
- Add tests on `inverse_transform` method of transforms on subset of segments, on new segments, on future with gap ([#1127](https://github.com/tinkoff-ai/etna/pull/1127))
- In-sample prediction for `BATSModel` and `TBATSModel` ([#1181](https://github.com/tinkoff-ai/etna/pull/1181))
- Method `predict_components` for forecast decomposition in `_TBATSAdapter` ([#1181](https://github.com/tinkoff-ai/etna/pull/1181))
- Forecast decomposition for `DeadlineMovingAverageModel`([#1186](https://github.com/tinkoff-ai/etna/pull/1186))
- Prediction decomposition example into `custom_transform_and_model.ipynb`([#1216](https://github.com/tinkoff-ai/etna/pull/1216))

### Changed
- Add optional `features` parameter in the signature of `TSDataset.to_pandas`, `TSDataset.to_flatten` ([#809](https://github.com/tinkoff-ai/etna/pull/809))
- Signature of the constructor of `TFTModel`, `DeepARModel` ([#1110](https://github.com/tinkoff-ai/etna/pull/1110))
- Interface of `Transform` and `PerSegmentWrapper` ([#835](https://github.com/tinkoff-ai/etna/pull/835))
- Signature of `TSDataset` methods `inverse_transform` and `make_future` now has `transforms` parameter. Remove transforms and regressors updating logic from TSDataset. Forecasts from the models are not internally inverse transformed. Methods `fit`,`transform`,`inverse_transform`  of `Transform` now works with `TSDataset` ([#956](https://github.com/tinkoff-ai/etna/pull/956))
- Create `AutoBase` and `AutoAbstract` classes, some of `Auto` class's logic moved there ([#1114](https://github.com/tinkoff-ai/etna/pull/1114)) 
- Impose specific order of columns on return value of `TSDataset.to_flatten` ([#1095](https://github.com/tinkoff-ai/etna/pull/1095))
- Add more scenarios into tests for models ([#1082](https://github.com/tinkoff-ai/etna/pull/1082))
- Decouple `SeasonalMovingAverageModel` from `PerSegmentModelMixin` ([#1132](https://github.com/tinkoff-ai/etna/pull/1132))
- Decouple `DeadlineMovingAverageModel` from `PerSegmentModelMixin` ([#1140](https://github.com/tinkoff-ai/etna/pull/1140))
- Remove version python-3.7 from `pyproject.toml`, update lock ([#1183](https://github.com/tinkoff-ai/etna/pull/1183))
- Bump minimum pandas version up to 1.1 ([#1214](https://github.com/tinkoff-ai/etna/pull/1214))

### Fixed
- Fix bug in `GaleShapleyFeatureSelectionTransform` with wrong number of remaining features ([#1110](https://github.com/tinkoff-ai/etna/pull/1110))
- `ProphetModel` fails with additional seasonality set ([#1157](https://github.com/tinkoff-ai/etna/pull/1157))
- Fix inference tests on new segments for `DeepARModel` and `TFTModel` ([#1109](https://github.com/tinkoff-ai/etna/pull/1109))
- Fix alignment during forecasting in new NNs, add validation of context size during forecasting in new NNs, add validation of batch in `MLPNet` ([#1108](https://github.com/tinkoff-ai/etna/pull/1108))
- Fix `MeanSegmentEncoderTransform` to work with subset of segments and raise error on new segments ([#1104](https://github.com/tinkoff-ai/etna/pull/1104))
- Fix outliers transforms on future with gap ([#1147](https://github.com/tinkoff-ai/etna/pull/1147))
- Fix `SegmentEncoderTransform` to work with subset of segments and raise error on new segments ([#1103](https://github.com/tinkoff-ai/etna/pull/1103))
- Fix `SklearnTransform` in per-segment mode to work on subset of segments and raise error on new segments ([#1107](https://github.com/tinkoff-ai/etna/pull/1107))
- Fix `OutliersTransform` and its children to raise error on new segments ([#1139](https://github.com/tinkoff-ai/etna/pull/1139))
- Fix `DifferencingTransform` to raise error on new segments during `transform` and `inverse_transform` in inplace mode ([#1141](https://github.com/tinkoff-ai/etna/pull/1141))
- Teach `DifferencingTransform` to `inverse_transform` with NaNs ([#1155](https://github.com/tinkoff-ai/etna/pull/1155))
- Fixed `custom_transform_and_model.ipynb`([#1216](https://github.com/tinkoff-ai/etna/pull/1216))

### Removed
- `sample_acf_plot`, `sample_pacf_plot`, `CatBoostModelPerSegment`, `CatBoostModelMultiSegment` ([#1118](https://github.com/tinkoff-ai/etna/pull/1118))
- `PytorchForecastingTransform` ([#971](https://github.com/tinkoff-ai/etna/pull/971))

## [1.15.0] - 2023-01-31
### Added
- `RMSE` metric & `rmse` functional metric ([#1051](https://github.com/tinkoff-ai/etna/pull/1051))
- `MaxDeviation` metric & `max_deviation` functional metric ([#1061](https://github.com/tinkoff-ai/etna/pull/1061))
- Add saving/loading for transforms, models, pipelines, ensembles; tutorial for saving/loading ([#1068](https://github.com/tinkoff-ai/etna/pull/1068))
- Add hierarchical time series support([#1083](https://github.com/tinkoff-ai/etna/pull/1083))
- Add `WAPE` metric & `wape` functional metric ([#1085](https://github.com/tinkoff-ai/etna/pull/1085))

### Fixed
- Missed kwargs in TFT init([#1078](https://github.com/tinkoff-ai/etna/pull/1078))

## [1.14.0] - 2022-12-16
### Added
- Add python 3.10 support ([#1005](https://github.com/tinkoff-ai/etna/pull/1005))
- Add `SumTranform`([#1021](https://github.com/tinkoff-ai/etna/pull/1021))
- Add `plot_change_points_interactive` ([#988](https://github.com/tinkoff-ai/etna/pull/988))
- Add `experimental` module with `TimeSeriesBinaryClassifier` and `PredictabilityAnalyzer` ([#985](https://github.com/tinkoff-ai/etna/pull/985))
- Inference track results: add `predict` method to pipelines, teach some models to work with context, change hierarchy of base models, update notebook examples ([#979](https://github.com/tinkoff-ai/etna/pull/979))
- Add `get_ruptures_regularization` into `experimental` module ([#1001](https://github.com/tinkoff-ai/etna/pull/1001))
- Add example `classification` notebook for experimental classification feature ([#997](https://github.com/tinkoff-ai/etna/pull/997)) 
### Changed
- Change returned model in get_model of BATSModel, TBATSModel ([#987](https://github.com/tinkoff-ai/etna/pull/987))
- Add acf_plot, deprecated sample_acf_plot, sample_pacf_plot ([#1004](https://github.com/tinkoff-ai/etna/pull/1004))
- Change returned model in `get_model` of `HoltWintersModel`, `HoltModel`, `SimpleExpSmoothingModel` ([#986](https://github.com/tinkoff-ai/etna/pull/986))
### Fixed
- Fix `MinMaxDifferenceTransform` import ([#1030](https://github.com/tinkoff-ai/etna/pull/1030))
- Fix release docs and docker images cron job ([#982](https://github.com/tinkoff-ai/etna/pull/982))
- Fix forecast first point with CatBoostPerSegmentModel ([#1010](https://github.com/tinkoff-ai/etna/pull/1010))
- Fix hanging EDA notebook ([#1027](https://github.com/tinkoff-ai/etna/pull/1027))
- Fix hanging EDA notebook v2 + cache clean script ([#1034](https://github.com/tinkoff-ai/etna/pull/1034))

## [1.13.0] - 2022-10-10
### Added
- Add `greater_is_better` property for Metric ([#921](https://github.com/tinkoff-ai/etna/pull/921))
- `etna.auto` for greedy search, `etna.auto.pool` with default pipelines, `etna.auto.optuna` wrapper for optuna ([#895](https://github.com/tinkoff-ai/etna/pull/895))
- Add `MinMaxDifferenceTransform` ([#955](https://github.com/tinkoff-ai/etna/pull/955))
- Add wandb sweeps and optuna examples ([#338](https://github.com/tinkoff-ai/etna/pull/338))
### Changed
- Make slicing faster in `TSDataset._merge_exog`, `FilterFeaturesTransform`, `AddConstTransform`, `LambdaTransform`, `LagTransform`, `LogTransform`, `SklearnTransform`, `WindowStatisticsTransform`; make CICD test different pandas versions ([#900](https://github.com/tinkoff-ai/etna/pull/900))
- Mark some tests as long ([#929](https://github.com/tinkoff-ai/etna/pull/929))
- Fix to_dict with nn models and add unsafe conversion for callbacks ([#949](https://github.com/tinkoff-ai/etna/pull/949))
### Fixed
- Fix `to_dict` with function as parameter ([#941](https://github.com/tinkoff-ai/etna/pull/941))
- Fix native networks to work with generated future equals to horizon ([#936](https://github.com/tinkoff-ai/etna/pull/936))
- Fix `SARIMAXModel` to work with exogenous data on `pmdarima>=2.0` ([#940](https://github.com/tinkoff-ai/etna/pull/940))
- Teach catboost to work with encoders ([#957](https://github.com/tinkoff-ai/etna/pull/957))
## [1.12.0] - 2022-09-05
### Added
- Function to transform etna objects to dict([#818](https://github.com/tinkoff-ai/etna/issues/818))
- `MLPModel`([#860](https://github.com/tinkoff-ai/etna/pull/860))
- `DeadlineMovingAverageModel` ([#827](https://github.com/tinkoff-ai/etna/pull/827))
- `DirectEnsemble` ([#824](https://github.com/tinkoff-ai/etna/pull/824))
- CICD: untaged docker image cleaner ([#856](https://github.com/tinkoff-ai/etna/pull/856))
- Notebook about forecasting strategies ([#864](https://github.com/tinkoff-ai/etna/pull/863))
- Add `ChangePointSegmentationTransform`, `RupturesChangePointsModel` ([#821](https://github.com/tinkoff-ai/etna/issues/821))
### Changed
- Teach AutoARIMAModel to work with out-sample predictions ([#830](https://github.com/tinkoff-ai/etna/pull/830))
- Make TSDataset.to_flatten faster for big datasets ([#848](https://github.com/tinkoff-ai/etna/pull/848))
### Fixed
- Type hints for external users by [PEP 561](https://mypy.readthedocs.io/en/stable/running_mypy.html#missing-library-stubs-or-py-typed-marker) ([#868](https://github.com/tinkoff-ai/etna/pull/868))
- Type hints for `Pipeline.model` match `models.nn`([#768](https://github.com/tinkoff-ai/etna/pull/840))
- Fix behavior of SARIMAXModel if simple_differencing=True is set ([#837](https://github.com/tinkoff-ai/etna/pull/837))
- Bug python3.7 and TypedDict import ([867](https://github.com/tinkoff-ai/etna/pull/867))
- Fix deprecated  pytorch lightning trainer flags ([#866](https://github.com/tinkoff-ai/etna/pull/866))
- ProphetModel doesn't work with cap and floor regressors ([#842](https://github.com/tinkoff-ai/etna/pull/842))
- Fix problem with encoding category types in OHE ([#843](https://github.com/tinkoff-ai/etna/pull/843))
- Change Docker cuda image version from 11.1 to 11.6.2 ([#838](https://github.com/tinkoff-ai/etna/pull/838))
- Optimize time complexity of `determine_num_steps`([#864](https://github.com/tinkoff-ai/etna/pull/864))
- All warning as errors([#880](https://github.com/tinkoff-ai/etna/pull/880))
- Update .gitignore with .DS_Store and checkpoints ([#883](https://github.com/tinkoff-ai/etna/pull/883))
- Delete ROADMAP.md ([#904]https://github.com/tinkoff-ai/etna/pull/904)
- Fix ci invalid cache ([#896](https://github.com/tinkoff-ai/etna/pull/896))

## [1.11.1] - 2022-08-03
### Fixed
- Fix missing `constant_value` in `TimeSeriesImputerTransform` ([#819](https://github.com/tinkoff-ai/etna/pull/819))
- Make in-sample predictions of SARIMAXModel non-dynamic in all cases ([#812](https://github.com/tinkoff-ai/etna/pull/812))
- Add known_future to cli docs ([#823](https://github.com/tinkoff-ai/etna/pull/823))

## [1.11.0] - 2022-07-25
### Added
- LSTM based RNN and native deep models base classes ([#776](https://github.com/tinkoff-ai/etna/pull/776))
- Lambda transform ([#762](https://github.com/tinkoff-ai/etna/issues/762))
- assemble pipelines ([#774](https://github.com/tinkoff-ai/etna/pull/774))
- Tests on in-sample, out-sample predictions with gap for all models ([#785](https://github.com/tinkoff-ai/etna/pull/786))
### Changed
- Add columns and mode parameters in plot_correlation_matrix ([#726](https://github.com/tinkoff-ai/etna/pull/753))
- Add CatBoostPerSegmentModel and CatBoostMultiSegmentModel classes, deprecate CatBoostModelPerSegment and CatBoostModelMultiSegment ([#779](https://github.com/tinkoff-ai/etna/pull/779))
- Allow Prophet update to 1.1 ([#799](https://github.com/tinkoff-ai/etna/pull/799))
- Make LagTransform, LogTransform, AddConstTransform vectorized ([#756](https://github.com/tinkoff-ai/etna/pull/756))
- Improve the behavior of plot_feature_relevance visualizing p-values ([#795](https://github.com/tinkoff-ai/etna/pull/795))
- Update poetry.core version ([#780](https://github.com/tinkoff-ai/etna/pull/780))
- Make native prediction intervals for DeepAR ([#761](https://github.com/tinkoff-ai/etna/pull/761))
- Make native prediction intervals for TFTModel ([#770](https://github.com/tinkoff-ai/etna/pull/770))
- Test cases for testing inference of models ([#794](https://github.com/tinkoff-ai/etna/pull/794))
- Wandb.log to WandbLogger ([#816](https://github.com/tinkoff-ai/etna/pull/816))
### Fixed
- Fix missing prophet in docker images ([#767](https://github.com/tinkoff-ai/etna/pull/767))
- Add `known_future` parameter to CLI ([#758](https://github.com/tinkoff-ai/etna/pull/758))
- FutureWarning: The frame.append method is deprecated. Use pandas.concat instead ([#764](https://github.com/tinkoff-ai/etna/pull/764))
- Correct ordering if multi-index in backtest ([#771](https://github.com/tinkoff-ai/etna/pull/771))
- Raise errors in models.nn if they can't make in-sample and some cases out-sample predictions ([#813](https://github.com/tinkoff-ai/etna/pull/813))
- Teach BATS/TBATS to work with in-sample, out-sample predictions correctly ([#806](https://github.com/tinkoff-ai/etna/pull/806))
- Github actions cache issue with poetry update ([#778](https://github.com/tinkoff-ai/etna/pull/778))

## [1.10.0] - 2022-06-12
### Added
- Add Sign metric ([#730](https://github.com/tinkoff-ai/etna/pull/730))
- Add AutoARIMA model ([#679](https://github.com/tinkoff-ai/etna/pull/679))
- Add parameters `start`, `end` to some eda methods ([#665](https://github.com/tinkoff-ai/etna/pull/665))
- Add BATS and TBATS model adapters ([#678](https://github.com/tinkoff-ai/etna/pull/734))
- Jupyter extension for black ([#742](https://github.com/tinkoff-ai/etna/pull/742))
### Changed
- Change color of lines in plot_anomalies and plot_clusters, add grid to all plots, make trend line thicker in plot_trend ([#705](https://github.com/tinkoff-ai/etna/pull/705))
- Change format of holidays for holiday_plot ([#708](https://github.com/tinkoff-ai/etna/pull/708))
- Make feature selection transforms return columns in inverse_transform([#688](https://github.com/tinkoff-ai/etna/issues/688))
- Add xticks parameter for plot_periodogram, clip frequencies to be >= 1 ([#706](https://github.com/tinkoff-ai/etna/pull/706))
- Make TSDataset method to_dataset work with copy of the passed dataframe ([#741](https://github.com/tinkoff-ai/etna/pull/741))
### Fixed
- Fix bug when `ts.plot` does not save figure ([#714](https://github.com/tinkoff-ai/etna/pull/714))
- Fix bug in plot_clusters ([#675](https://github.com/tinkoff-ai/etna/pull/675))
- Fix bugs and documentation for cross_corr_plot ([#691](https://github.com/tinkoff-ai/etna/pull/691))
- Fix bugs and documentation for plot_backtest and plot_backtest_interactive ([#700](https://github.com/tinkoff-ai/etna/pull/700))
- Make STLTransform to work with NaNs at the beginning ([#736](https://github.com/tinkoff-ai/etna/pull/736))
- Fix tiny prediction intervals ([#722](https://github.com/tinkoff-ai/etna/pull/722))
- Fix deepcopy issue for fitted deepmodel ([#735](https://github.com/tinkoff-ai/etna/pull/735))
- Fix making backtest if all segments start with NaNs ([#728](https://github.com/tinkoff-ai/etna/pull/728))
- Fix logging issues with backtest while emp intervals using ([#747](https://github.com/tinkoff-ai/etna/pull/747))

## [1.9.0] - 2022-05-17
### Added
- Add plot_metric_per_segment ([#658](https://github.com/tinkoff-ai/etna/pull/658))
- Add metric_per_segment_distribution_plot ([#666](https://github.com/tinkoff-ai/etna/pull/666))
### Changed
- Remove parameter normalize in linear models ([#686](https://github.com/tinkoff-ai/etna/pull/686))
### Fixed
- Add missed `forecast_params` in forecast CLI method ([#671](https://github.com/tinkoff-ai/etna/pull/671))
- Add `_per_segment_average` method to the Metric class ([#684](https://github.com/tinkoff-ai/etna/pull/684))
- Fix `get_statistics_relevance_table` working with NaNs and categoricals ([#672](https://github.com/tinkoff-ai/etna/pull/672))
- Fix bugs and documentation for stl_plot ([#685](https://github.com/tinkoff-ai/etna/pull/685))
- Fix cuda docker images ([#694](https://github.com/tinkoff-ai/etna/pull/694)])

## [1.8.0] - 2022-04-28
### Added
- `Width` and `Coverage` metrics for prediction intervals ([#638](https://github.com/tinkoff-ai/etna/pull/638))
- Masked backtest ([#613](https://github.com/tinkoff-ai/etna/pull/613))
- Add seasonal_plot ([#628](https://github.com/tinkoff-ai/etna/pull/628))
- Add plot_periodogram ([#606](https://github.com/tinkoff-ai/etna/pull/606))
- Add support of quantiles in backtest ([#652](https://github.com/tinkoff-ai/etna/pull/652))
- Add prediction_actual_scatter_plot ([#610](https://github.com/tinkoff-ai/etna/pull/610))
- Add plot_holidays ([#624](https://github.com/tinkoff-ai/etna/pull/624))
- Add instruction about documentation formatting to contribution guide ([#648](https://github.com/tinkoff-ai/etna/pull/648))
- Seasonal strategy in TimeSeriesImputerTransform ([#639](https://github.com/tinkoff-ai/etna/pull/639))

### Changed
- Add logging to `Metric.__call__` ([#643](https://github.com/tinkoff-ai/etna/pull/643))
- Add in_column to plot_anomalies, plot_anomalies_interactive ([#618](https://github.com/tinkoff-ai/etna/pull/618))
- Add logging to TSDataset.inverse_transform ([#642](https://github.com/tinkoff-ai/etna/pull/642))

### Fixed
- Passing non default params for default models STLTransform ([#641](https://github.com/tinkoff-ai/etna/pull/641))
- Fixed bug in SARIMAX model with `horizon`=1 ([#637](https://github.com/tinkoff-ai/etna/pull/637))
- Fixed bug in models `get_model` method ([#623](https://github.com/tinkoff-ai/etna/pull/623))
- Fixed unsafe comparison in plots ([#611](https://github.com/tinkoff-ai/etna/pull/611))
- Fixed plot_trend does not work with Linear and TheilSen transforms ([#617](https://github.com/tinkoff-ai/etna/pull/617))
- Improve computation time for rolling window statistics ([#625](https://github.com/tinkoff-ai/etna/pull/625))
- Don't fill first timestamps in TimeSeriesImputerTransform ([#634](https://github.com/tinkoff-ai/etna/pull/634))
- Fix documentation formatting ([#636](https://github.com/tinkoff-ai/etna/pull/636))
- Fix bug with exog features in AutoRegressivePipeline ([#647](https://github.com/tinkoff-ai/etna/pull/647))
- Fix missed dependencies ([#656](https://github.com/tinkoff-ai/etna/pull/656))
- Fix custom_transform_and_model notebook ([#651](https://github.com/tinkoff-ai/etna/pull/651))
- Fix MyBinder bug with dependencies ([#650](https://github.com/tinkoff-ai/etna/pull/650))

## [1.7.0] - 2022-03-16
### Added
- Regressors logic to TSDatasets init ([#357](https://github.com/tinkoff-ai/etna/pull/357))
- `FutureMixin` into some transforms ([#361](https://github.com/tinkoff-ai/etna/pull/361))
- Regressors updating in TSDataset transform loops ([#374](https://github.com/tinkoff-ai/etna/pull/374))
- Regressors handling in TSDataset `make_future` and `train_test_split` ([#447](https://github.com/tinkoff-ai/etna/pull/447))
- Prediction intervals visualization in `plot_forecast` ([#538](https://github.com/tinkoff-ai/etna/pull/538))
- Add plot_imputation ([#598](https://github.com/tinkoff-ai/etna/pull/598))
- Add plot_time_series_with_change_points function ([#534](https://github.com/tinkoff-ai/etna/pull/534))
- Add plot_trend ([#565](https://github.com/tinkoff-ai/etna/pull/565))
- Add find_change_points function ([#521](https://github.com/tinkoff-ai/etna/pull/521))
- Add option `day_number_in_year` to DateFlagsTransform ([#552](https://github.com/tinkoff-ai/etna/pull/552))
- Add plot_residuals ([#539](https://github.com/tinkoff-ai/etna/pull/539))
- Add get_residuals ([#597](https://github.com/tinkoff-ai/etna/pull/597))
- Create `PerSegmentBaseModel`, `PerSegmentPredictionIntervalModel` ([#537](https://github.com/tinkoff-ai/etna/pull/537))
- Create `MultiSegmentModel` ([#551](https://github.com/tinkoff-ai/etna/pull/551))
- Add qq_plot ([#604](https://github.com/tinkoff-ai/etna/pull/604))
- Add regressors example notebook ([#577](https://github.com/tinkoff-ai/etna/pull/577))
- Create `EnsembleMixin` ([#574](https://github.com/tinkoff-ai/etna/pull/574))
- Add option `season_number` to DateFlagsTransform ([#567](https://github.com/tinkoff-ai/etna/pull/567))
- Create `BasePipeline`, add prediction intervals to all the pipelines, move parameter n_fold to forecast ([#578](https://github.com/tinkoff-ai/etna/pull/578))
- Add stl_plot ([#575](https://github.com/tinkoff-ai/etna/pull/575))
- Add plot_features_relevance ([#579](https://github.com/tinkoff-ai/etna/pull/579))
- Add community section to README.md ([#580](https://github.com/tinkoff-ai/etna/pull/580))
- Create `AbstaractPipeline` ([#573](https://github.com/tinkoff-ai/etna/pull/573))
- Option "auto" to `weights` parameter of `VotingEnsemble`, enables to use feature importance as weights of base estimators ([#587](https://github.com/tinkoff-ai/etna/pull/587)) 

### Changed
- Change the way `ProphetModel` works with regressors ([#383](https://github.com/tinkoff-ai/etna/pull/383))
- Change the way `SARIMAXModel` works with regressors ([#380](https://github.com/tinkoff-ai/etna/pull/380)) 
- Change the way `Sklearn` models works with regressors ([#440](https://github.com/tinkoff-ai/etna/pull/440))
- Change the way `FeatureSelectionTransform` works with regressors, rename variables replacing the "regressor" to "feature" ([#522](https://github.com/tinkoff-ai/etna/pull/522))
- Add table option to ConsoleLogger ([#544](https://github.com/tinkoff-ai/etna/pull/544))
- Installation instruction ([#526](https://github.com/tinkoff-ai/etna/pull/526))
- Update plot_forecast for multi-forecast mode ([#584](https://github.com/tinkoff-ai/etna/pull/584))
- Trainer kwargs for deep models ([#540](https://github.com/tinkoff-ai/etna/pull/540))
- Update CONTRIBUTING.md ([#536](https://github.com/tinkoff-ai/etna/pull/536))
- Rename `_CatBoostModel`, `_HoltWintersModel`, `_SklearnModel` ([#543](https://github.com/tinkoff-ai/etna/pull/543))
- Add logging to TSDataset.make_future, log repr of transform instead of class name ([#555](https://github.com/tinkoff-ai/etna/pull/555))
- Rename `_SARIMAXModel` and `_ProphetModel`, make `SARIMAXModel` and `ProphetModel` inherit from `PerSegmentPredictionIntervalModel` ([#549](https://github.com/tinkoff-ai/etna/pull/549))
- Update get_started section in README ([#569](https://github.com/tinkoff-ai/etna/pull/569))
- Make detrending polynomial ([#566](https://github.com/tinkoff-ai/etna/pull/566))
- Update documentation about transforms that generate regressors, update examples with them ([#572](https://github.com/tinkoff-ai/etna/pull/572))
- Fix that segment is string ([#602](https://github.com/tinkoff-ai/etna/pull/602))
- Make `LabelEncoderTransform` and `OneHotEncoderTransform` multi-segment ([#554](https://github.com/tinkoff-ai/etna/pull/554))

### Fixed
- Fix `TSDataset._update_regressors` logic removing the regressors ([#489](https://github.com/tinkoff-ai/etna/pull/489)) 
- Fix `TSDataset.info`, `TSDataset.describe` methods ([#519](https://github.com/tinkoff-ai/etna/pull/519))
- Fix regressors handling for `OneHotEncoderTransform` and `HolidayTransform` ([#518](https://github.com/tinkoff-ai/etna/pull/518))
- Fix wandb summary issue with custom plots ([#535](https://github.com/tinkoff-ai/etna/pull/535))
- Small notebook fixes ([#595](https://github.com/tinkoff-ai/etna/pull/595))
- Fix import Literal in plotters ([#558](https://github.com/tinkoff-ai/etna/pull/558))
- Fix plot method bug when plot method does not plot all required segments ([#596](https://github.com/tinkoff-ai/etna/pull/596))
- Fix dependencies for ARM ([#599](https://github.com/tinkoff-ai/etna/pull/599))
- [BUG] nn models make forecast without inverse_transform ([#541](https://github.com/tinkoff-ai/etna/pull/541))

## [1.6.3] - 2022-02-14

### Fixed

- Fixed adding unnecessary lag=1 in statistics ([#523](https://github.com/tinkoff-ai/etna/pull/523))
- Fixed wrong MeanTransform behaviour when using alpha parameter ([#523](https://github.com/tinkoff-ai/etna/pull/523))
- Fix processing add_noise=True parameter in datasets generation ([#520](https://github.com/tinkoff-ai/etna/pull/520))
- Fix scipy version ([#525](https://github.com/tinkoff-ai/etna/pull/525))

## [1.6.2] - 2022-02-09
### Added
- Holt-Winters', Holt and exponential smoothing models ([#502](https://github.com/tinkoff-ai/etna/pull/502))

### Fixed
- Bug with exog features in DifferencingTransform.inverse_transform ([#503](https://github.com/tinkoff-ai/etna/pull/503))

## [1.6.1] - 2022-02-03
### Added
- Allow choosing start and end in `TSDataset.plot` method ([488](https://github.com/tinkoff-ai/etna/pull/488))

### Changed
- Make TSDataset.to_flatten faster ([#475](https://github.com/tinkoff-ai/etna/pull/475))
- Allow logger percentile metric aggregation to work with NaNs ([#483](https://github.com/tinkoff-ai/etna/pull/483))

### Fixed
- Can't make forecasting with pipelines, data with nans, and Imputers ([#473](https://github.com/tinkoff-ai/etna/pull/473))


## [1.6.0] - 2022-01-28

### Added

- Method TSDataset.info ([#409](https://github.com/tinkoff-ai/etna/pull/409))
- DifferencingTransform ([#414](https://github.com/tinkoff-ai/etna/pull/414))
- OneHotEncoderTransform and LabelEncoderTransform ([#431](https://github.com/tinkoff-ai/etna/pull/431))
- MADTransform ([#441](https://github.com/tinkoff-ai/etna/pull/441))
- `MRMRFeatureSelectionTransform` ([#439](https://github.com/tinkoff-ai/etna/pull/439))
- Possibility to change metric representation in backtest using Metric.name ([#454](https://github.com/tinkoff-ai/etna/pull/454))
- Warning section in documentation about look-ahead bias ([#464](https://github.com/tinkoff-ai/etna/pull/464))
- Parameter `figsize` to all the plotters [#465](https://github.com/tinkoff-ai/etna/pull/465)

### Changed

- Change method TSDataset.describe ([#409](https://github.com/tinkoff-ai/etna/pull/409))
- Group Transforms according to their impact ([#420](https://github.com/tinkoff-ai/etna/pull/420))
- Change the way `LagTransform`, `DateFlagsTransform` and `TimeFlagsTransform` generate column names ([#421](https://github.com/tinkoff-ai/etna/pull/421))
- Clarify the behaviour of TimeSeriesImputerTransform in case of all NaN values ([#427](https://github.com/tinkoff-ai/etna/pull/427))
- Fixed bug in title in `sample_acf_plot` method ([#432](https://github.com/tinkoff-ai/etna/pull/432))
- Pytorch-forecasting and sklearn version update + some pytroch transform API changing ([#445](https://github.com/tinkoff-ai/etna/pull/445))

### Fixed

- Add relevance_params in GaleShapleyFeatureSelectionTransform ([#410](https://github.com/tinkoff-ai/etna/pull/410))
- Docs for statistics transforms ([#441](https://github.com/tinkoff-ai/etna/pull/441))
- Handling NaNs in trend transforms ([#456](https://github.com/tinkoff-ai/etna/pull/456))
- Logger fails with StackingEnsemble ([#460](https://github.com/tinkoff-ai/etna/pull/460))
- SARIMAX parameters fix ([#459](https://github.com/tinkoff-ai/etna/pull/459))
- [BUG] Check pytorch-forecasting models with freq > "1D" ([#463](https://github.com/tinkoff-ai/etna/pull/463))

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
