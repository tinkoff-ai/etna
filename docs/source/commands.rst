CLI commands
=============

Basic ``forecast`` usage:
-------------------------

.. code-block:: console

        Usage: etna forecast [OPTIONS] CONFIG_PATH TARGET_PATH FREQ OUTPUT_PATH [EXOG_PATH]
                             [FORECAST_CONFIG_PATH] [RAW_OUTPUT] [KNOWN_FUTURE]

        Command to make forecast with etna without coding.

        Arguments:
            CONFIG_PATH             path to yaml config with desired pipeline  [required]
            TARGET_PATH             path to csv with data to forecast  [required]
            FREQ                    frequency of timestamp in files in pandas format  [required]
            OUTPUT_PATH             where to save forecast  [required]
            [EXOG_PATH]             path to csv with exog data
            [FORECAST_CONFIG_PATH]  path to yaml config with forecast params
            [RAW_OUTPUT]            by default we return only forecast without features [default: False]
            [KNOWN_FUTURE]          list of all known_future columns (regressor columns). If not specified then all exog_columns considered known_future [default: None]

**Forecast config parameters**

* :code:`prediction_interval` - whether to estimate prediction interval for forecast.
* :code:`quantiles` - levels of prediction distribution. By default 2.5% and 97.5% are taken to form a 95% prediction interval.
* :code:`n_folds` - number of folds to use in the backtest for prediction interval estimation. By default equals to 3.
* :code:`return_components` - whether to estimate forecast components
* :code:`start_timestamp` - timestamp with the starting point of forecast.

Setting these parameters is optional.
Further information on arguments could be found in the documentation of :meth:`~etna.pipeline.pipeline.Pipeline.forecast` method.

**How to create config?**

Example of pipeline's config:

.. code-block:: yaml

    _target_: etna.pipeline.Pipeline
    horizon: 4
    model:
      _target_: etna.models.CatBoostMultiSegmentModel
    transforms:
      - _target_: etna.transforms.LinearTrendTransform
        in_column: target
      - _target_: etna.transforms.SegmentEncoderTransform

Example of forecast params config:

.. code-block:: yaml

    prediction_interval: true
    quantiles: [0.025, 0.975]
    n_folds: 3

Parameter :code:`start_timestamp` could be set similarly:

.. code-block:: yaml

    prediction_interval: true
    quantiles: [0.025, 0.975]
    start_timestamp: "2020-01-12"

**How to prepare data?**

Example of dataset with data to forecast:

=============  ===========  ==========
  timestamp      segment      target
=============  ===========  ==========
2020-01-01     segment_1         1
2020-01-02     segment_1         2
2020-01-03     segment_1         3
2020-01-04     segment_1         4
...
2020-01-10     segment_2        10
2020-01-11     segment_2        20
=============  ===========  ==========

Example of exog dataset:

=============  ===========  ===============  ===============
  timestamp      segment      regressor_1      regressor_2
=============  ===========  ===============  ===============
2020-01-01     segment_1          11               12
2020-01-02     segment_1          22               13
2020-01-03     segment_1          31               14
2020-01-04     segment_1          42               15
...
2020-02-10     segment_2         101               61
2020-02-11     segment_2         205               54
=============  ===========  ===============  ===============

---------------------------


Basic ``backtest`` usage:
--------------------------

.. code-block:: console

        Usage: etna backtest [OPTIONS] CONFIG_PATH BACKTEST_CONFIG_PATH TARGET_PATH FREQ OUTPUT_PATH [EXOG_PATH] [KNOWN_FUTURE]

        Command to run backtest with etna without coding.

        Arguments:
            CONFIG_PATH             path to yaml config with desired pipeline  [required]
            BACKTEST_CONFIG_PATH    path to yaml with backtest run config [required]
            TARGET_PATH             path to csv with data to forecast  [required]
            FREQ                    frequency of timestamp in files in pandas format  [required]
            OUTPUT_PATH             where to save forecast  [required]
            [EXOG_PATH]             path to csv with exog data
            [KNOWN_FUTURE]          list of all known_future columns (regressor columns). If not specified then all exog_columns considered known_future [default: None]


**How to create configs?**

Example of pipeline's config:

.. code-block:: yaml

    _target_: etna.pipeline.Pipeline
    horizon: 4
    model:
      _target_: etna.models.CatBoostMultiSegmentModel
    transforms:
      - _target_: etna.transforms.LinearTrendTransform
        in_column: target
      - _target_: etna.transforms.SegmentEncoderTransform

Example of backtest run config:

.. code-block:: yaml

    n_folds: 3
    n_jobs: 3
    metrics:
      - _target_: etna.metrics.MAE
      - _target_: etna.metrics.MSE
      - _target_: etna.metrics.MAPE
      - _target_: etna.metrics.SMAPE


**How to prepare data?**

Example of dataset with data to forecast:

=============  ===========  ==========
  timestamp      segment      target
=============  ===========  ==========
2020-01-01     segment_1         1
2020-01-02     segment_1         2
2020-01-03     segment_1         3
2020-01-04     segment_1         4
...
2020-01-10     segment_2        10
2020-01-11     segment_2        20
=============  ===========  ==========

Example of exog dataset:

=============  ===========  ===============  ===============
  timestamp      segment      regressor_1      regressor_2
=============  ===========  ===============  ===============
2020-01-01     segment_1          11               12
2020-01-02     segment_1          22               13
2020-01-03     segment_1          31               14
2020-01-04     segment_1          42               15
...
2020-02-10     segment_2         101               61
2020-02-11     segment_2         205               54
=============  ===========  ===============  ===============
