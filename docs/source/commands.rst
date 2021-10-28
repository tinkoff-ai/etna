CLI commands
========

Basic ``forecast`` usage:

.. code-block:: console

        Usage: etna forecast [OPTIONS] CONFIG_PATH TARGET_PATH FREQ OUTPUT_PATH [EXOG_PATH]
                    [RAW_OUTPUT]

        Command to make forecast with etna without coding.

        Expected format of csv with target timeseries:

        | timestamp           | segment   |   target |
        |:--------------------|:----------|---------:|
        | 2019-01-01 00:00:00 | segment_a |      170 |
        | 2019-01-02 00:00:00 | segment_a |      243 |
        | 2019-01-03 00:00:00 | segment_a |      267 |
        | 2019-01-04 00:00:00 | segment_a |      287 |
        | 2019-01-05 00:00:00 | segment_a |      279 |

        Expected format of csv with exogenous timeseries:

        | timestamp           |   regressor_1 |   regressor_2 | segment   |
        |:--------------------|--------------:|--------------:|:----------|
        | 2019-01-01 00:00:00 |             0 |             5 | segment_a |
        | 2019-01-02 00:00:00 |             1 |             6 | segment_a |
        | 2019-01-03 00:00:00 |             2 |             7 | segment_a |
        | 2019-01-04 00:00:00 |             3 |             8 | segment_a |
        | 2019-01-05 00:00:00 |             4 |             9 | segment_a |

        Arguments:
            CONFIG_PATH   path to yaml config with desired pipeline  [required]
            TARGET_PATH   path to csv with data to forecast  [required]
            FREQ          frequency of timestamp in files in pandas format  [required]
            OUTPUT_PATH   where to save forecast  [required]
            [EXOG_PATH]   path to csv with exog data
            [RAW_OUTPUT]  by default we return only forecast without features  [default: False]


Basic ``backtest`` usage:

.. code-block:: console

        Usage: etna backtest forecast CONFIG_PATH BACKTEST_CONFIG_PATH TARGET_PATH FREQ OUTPUT_PATH [EXOG_PATH]

        Command to run backtest with etna without coding.

        Expected format of csv with target timeseries:

        | timestamp           | segment   |   target |
        |:--------------------|:----------|---------:|
        | 2019-01-01 00:00:00 | segment_a |      170 |
        | 2019-01-02 00:00:00 | segment_a |      243 |
        | 2019-01-03 00:00:00 | segment_a |      267 |
        | 2019-01-04 00:00:00 | segment_a |      287 |
        | 2019-01-05 00:00:00 | segment_a |      279 |

        Expected format of csv with exogenous timeseries:

        | timestamp           |   regressor_1 |   regressor_2 | segment   |
        |:--------------------|--------------:|--------------:|:----------|
        | 2019-01-01 00:00:00 |             0 |             5 | segment_a |
        | 2019-01-02 00:00:00 |             1 |             6 | segment_a |
        | 2019-01-03 00:00:00 |             2 |             7 | segment_a |
        | 2019-01-04 00:00:00 |             3 |             8 | segment_a |
        | 2019-01-05 00:00:00 |             4 |             9 | segment_a |

        Arguments:
            CONFIG_PATH             path to yaml config with desired pipeline  [required]
            BACKTEST_CONFIG_PATH    path to yaml with backtest run config [required]
            TARGET_PATH             path to csv with data to forecast  [required]
            FREQ                    frequency of timestamp in files in pandas format  [required]
            OUTPUT_PATH             where to save forecast  [required]
            [EXOG_PATH]             path to csv with exog data

.. _commands:

.. currentmodule:: etna

Details of ETNA CLI
-------------------------

See the API documentation for further details on ETNA commands:

.. currentmodule:: etna

.. moduleautosummary::
   :toctree: api/
   :template: custom-module-template.rst
   :recursive:

   etna.commands