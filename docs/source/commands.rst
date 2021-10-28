CLI commands
========

Basic ``forecast`` usage:

.. code-block:: console

        Usage: etna forecast [OPTIONS] CONFIG_PATH TARGET_PATH FREQ OUTPUT_PATH [EXOG_PATH]
                    [RAW_OUTPUT]

        Command to make forecast with etna without coding.

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