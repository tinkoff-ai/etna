.. _models:

Models
======

.. automodule:: etna.models
    :no-members:
    :no-inherited-members:

API details
-----------

.. currentmodule:: etna.models

Base:

.. autosummary::
   :toctree: api/
   :template: class.rst

   NonPredictionIntervalContextIgnorantAbstractModel
   NonPredictionIntervalContextRequiredAbstractModel
   PredictionIntervalContextIgnorantAbstractModel
   PredictionIntervalContextRequiredAbstractModel

Naive models:

.. autosummary::
   :toctree: api/
   :template: class.rst

   SeasonalMovingAverageModel
   MovingAverageModel
   NaiveModel
   DeadlineMovingAverageModel

Statistical models:

.. autosummary::
   :toctree: api/
   :template: class.rst

   AutoARIMAModel
   SARIMAXModel
   HoltWintersModel
   HoltModel
   SimpleExpSmoothingModel
   ProphetModel
   TBATSModel
   BATSModel
   StatsForecastARIMAModel
   StatsForecastAutoARIMAModel
   StatsForecastAutoCESModel
   StatsForecastAutoETSModel
   StatsForecastAutoThetaModel

ML-models:

.. autosummary::
   :toctree: api/
   :template: class.rst

   CatBoostMultiSegmentModel
   CatBoostPerSegmentModel
   ElasticMultiSegmentModel
   ElasticPerSegmentModel
   LinearMultiSegmentModel
   LinearPerSegmentModel
   SklearnMultiSegmentModel
   SklearnPerSegmentModel

The following models are neural networks.

.. note::
    Module ``etna.models.nn`` requires ``torch`` extension to be installed.
    Read more about this at :ref:`installation instruction <installation>`.

Native neural network models:

.. autosummary::
   :toctree: api/
   :template: class.rst

   nn.RNNModel
   nn.MLPModel
   nn.DeepStateModel
   nn.NBeatsGenericModel
   nn.NBeatsInterpretableModel
   nn.PatchTSModel

Neural network models based on :code:`pytorch_forecasting`:

.. autosummary::
   :toctree: api/
   :template: class.rst

   nn.DeepARModel
   nn.TFTModel

Utilities for neural network models based on :code:`pytorch_forecasting`:

.. autosummary::
   :toctree: api/
   :template: class.rst

   nn.PytorchForecastingDatasetBuilder
