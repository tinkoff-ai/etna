.. _analysis:

Analysis
===========

.. automodule:: etna.analysis
    :no-members:
    :no-inherited-members:

API details
-----------

See the API documentation for further details on available analysis tools:

.. currentmodule:: etna.analysis

Decomposition analysis utilities:

.. autosummary::
   :toctree: api/
   :template: base.rst

   find_change_points
   plot_change_points_interactive
   plot_time_series_with_change_points
   plot_trend
   seasonal_plot
   stl_plot

.. autosummary::
   :toctree: api/
   :template: class.rst

   SeasonalPlotAggregation
   SeasonalPlotAlignment
   SeasonalPlotCycle

EDA utilities:

.. autosummary::
   :toctree: api/
   :template: base.rst

   acf_plot
   cross_corr_plot
   distribution_plot
   get_correlation_matrix
   plot_clusters
   plot_correlation_matrix
   plot_holidays
   plot_imputation
   plot_periodogram

Feature selection analysis utilities:

.. autosummary::
   :toctree: api/
   :template: base.rst

   plot_feature_relevance
   ModelRelevanceTable
   RelevanceTable
   StatisticsRelevanceTable
   get_model_relevance_table
   get_statistics_relevance_table
   AggregationMode

Forecast analysis utilities:

.. autosummary::
   :toctree: api/
   :template: base.rst

   get_residuals
   metric_per_segment_distribution_plot
   plot_backtest
   plot_backtest_interactive
   plot_forecast
   plot_forecast_decomposition
   plot_metric_per_segment
   plot_residuals
   prediction_actual_scatter_plot
   qq_plot

.. autosummary::
   :toctree: api/
   :template: class.rst

   MetricPlotType
   PerFoldAggregation

Outliers analysis utilities:

.. autosummary::
   :toctree: api/
   :template: base.rst

   plot_anomalies
   plot_anomalies_interactive
   get_anomalies_density
   get_anomalies_hist
   get_anomalies_median
   get_anomalies_prediction_interval
