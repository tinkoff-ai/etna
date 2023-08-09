.. _glossary:

Glossary
========

This page lists some common terms used in documentation of the library.

.. glossary::

   Time series
      A series of variable measurements obtained at successive times according to :term:`frequency <time series frequency>`.

   Time series frequency
      Quantity that determines how often we take measurements for :term:`time series`.
      It doesn't have to be always the same number of seconds.
      For example, taking the first day of each month is a valid frequency.

   Univariate time series
      A single :term:`time series` containing measurements of a scalar variable.

   Multivariate time series
      A single :term:`time series` containing measurements of a multidimensional variable.

   Panel time series
      Multiple :term:`time series`. It is closely related to :term:`multivariate time series`,
      but the second term is usually used when the components are closely related,
      and it is more useful to treat them as a single multidimensional value.

   Hierarchical time series
      Multiple :term:`time series` having a level structure in which higher levels can be disaggregated
      by different attributes of interest into series of lower levels.
      See :doc:`tutorials/14-hierarchical_pipeline`.

   Segment
      We use this term to refer to one :term:`time series` in a dataset.

   Endogenous data
      Variables which measurements we want to model. It is often referred to as the "target".

   Exogenous data
      Additional variables in a dataset that help to model :term:`target <endogenous data>`.

   Regressor
      :term:`Exogenous variable <exogenous data>` whose values are known in the future during :term:`forecasting`.

   Stationarity
      Property of a time series to retain its statistical properties over time.

   Seasonality
      Property of time series to have a seasonal pattern of some fixed length.
      For example, weekly pattern for daily time series.

   Trend
      Property of time series to have a long-term change of the mean value.

   Change-point
      Point in a time series where its behavior changes.
      Its existence is the reason why you shouldn't trust your long-term forecasts too much.

   Forecasting
      The task of predicting future values of a time series.
      We are only interested in forecasting :term:`target <endogenous data>` variables.

   Forecasting horizon
      Set of time points we are going to :term:`forecast <forecasting>`. Often it is set to a fixed value.
      For example, horizon is equal to 7 if we want to make a forecast on 7 time points ahead for daily time series.

   Forecast confidence intervals
      Confidence intervals for the :math:`\mathop{E}(y | X)`.
      Set of intervals for every point in the :term:`horizon <forecasting horizon>` can be called a confidence band.
      Often confused with :term:`prediction intervals <forecast prediction intervals>`,
      see `The difference between prediction intervals and confidence intervals <https://robjhyndman.com/hyndsight/intervals/>`_ to understand the difference.

   Forecast prediction intervals
      Prediction intervals for predicted random variables.
      Set of intervals for every point in the :term:`horizon <forecasting horizon>` can be called a prediction band.
      Often confused with :term:`confidence intervals <forecast confidence intervals>`,
      see `The difference between prediction intervals and confidence intervals <https://robjhyndman.com/hyndsight/intervals/>`_ to understand the difference.

   Forecast prediction components
      In forecast decomposition each point is represented as the sum or product of some fixed terms. These terms are called components.

   Backtesting
      Type of cross-validation when we check the quality of the forecast model using historical data.

   Per-segment / local approach
      Mode of operation when there is a separate :term:`model` / :term:`transform` for each :term:`segment` of the dataset.

   Multi-segment / Global approach
      Mode of operation when there is one :term:`model` / :term:`transform` for every :term:`segment` of the dataset.

   Forecasting strategy
      Algorithm for using an ML model to produce a multi-step time series :term:`forecast <forecasting>`.
      See :doc:`tutorials/09-forecasting_strategies`.

   Clustering
      The task of finding clusters of similar time series.

   Classification
      The task of predicting a categorical label for the whole time series.

   Segmentation
      The task of dividing each time series into sequence of intervals with different characteristics.
      These intervals are separated by :term:`change-points <change-point>`.
      This shouldn't be confused with the term :term:`segment`.

   Dataset
      Collection of time series to work with.
      In the context of the library this is often used to refer to :py:class:`~etna.datasets.tsdataset.TSDataset`.

   Model
      Entity for learning time series patterns to make a :term:`forecast <forecasting>`. See :doc:`api_reference/models`.

   Transform
      Entity for performing transformations on a :term:`dataset`. See :doc:`api_reference/transforms`.

   Pipeline
      High-level entity for solving :term:`forecasting` task. Works with :term:`dataset`, :term:`model`, :term:`transforms <transform>` and other :term:`pipelines <pipeline>`.

   Lags
      The features generated by :py:class:`~etna.transforms.math.lags.LagTransform`.

   Date flags
      The features generated by :py:class:`~etna.transforms.timestamp.date_flags.DateFlagsTransform`.

   Fourier terms
      The features generated by :py:class:`~etna.transforms.timestamp.fourier.FourierTransform`.

   Differencing
      Time series :term:`transformation <transform>` that takes the differences between consecutive time points.
      See :py:class:`~etna.transforms.math.differencing.DifferencingTransform`.
