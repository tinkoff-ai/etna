.. _experimental:

Experimental
============

API details
-----------

.. currentmodule:: etna.experimental

Change-point utilities:

.. autosummary::
   :toctree: api/
   :template: base.rst

   change_points.get_ruptures_regularization

Classification of time-series:

.. note::
    This module requires ``classification`` extension to be installed.
    Read more about this at :ref:`installation instruction <installation>`.

.. autosummary::
   :toctree: api/
   :template: class.rst

   classification.TimeSeriesBinaryClassifier
   classification.PredictabilityAnalyzer
   classification.feature_extraction.TSFreshFeatureExtractor
   classification.feature_extraction.WEASELFeatureExtractor
