.. _transforms:

Transforms
==========

.. automodule:: etna.transforms
    :no-members:
    :no-inherited-members:

API details
-----------

.. currentmodule:: etna.transforms

Base classes:

.. autosummary::
   :toctree: api/
   :template: class.rst

   Transform
   IrreversibleTransform
   ReversibleTransform
   PerSegmentWrapper
   IrreversiblePerSegmentWrapper
   ReversiblePerSegmentWrapper
   OneSegmentTransform

Decomposition transforms:

.. autosummary::
   :toctree: api/
   :template: class.rst

   ChangePointsLevelTransform
   ChangePointsSegmentationTransform
   ChangePointsTrendTransform
   DeseasonalityTransform
   LinearTrendTransform
   STLTransform
   TheilSenTrendTransform
   TrendTransform
   ChangePointsLevelTransform
   ChangePointsSegmentationTransform
   ChangePointsTrendTransform
   DeseasonalityTransform
   LinearTrendTransform
   STLTransform
   TheilSenTrendTransform
   TrendTransform

Categorical encoding transforms:

.. autosummary::
   :toctree: api/
   :template: class.rst

   SegmentEncoderTransform
   MeanSegmentEncoderTransform
   LabelEncoderTransform
   OneHotEncoderTransform

Feature selection transforms:

.. autosummary::
   :toctree: api/
   :template: class.rst

   FilterFeaturesTransform
   TreeFeatureSelectionTransform
   GaleShapleyFeatureSelectionTransform
   MRMRFeatureSelectionTransform

Transforms to work with missing values:

.. autosummary::
   :toctree: api/
   :template: class.rst

   TimeSeriesImputerTransform
   ResampleWithDistributionTransform

Transforms to detect outliers:

.. autosummary::
   :toctree: api/
   :template: class.rst

   DensityOutliersTransform
   MedianOutliersTransform
   PredictionIntervalOutliersTransform

Transforms to work with time-related features:

.. autosummary::
   :toctree: api/
   :template: class.rst

   DateFlagsTransform
   TimeFlagsTransform
   SpecialDaysTransform
   HolidayTransform
   FourierTransform

Shift transforms:

.. autosummary::
   :toctree: api/
   :template: class.rst

   LagTransform
   ExogShiftTransform

Window-based transforms:

.. autosummary::
   :toctree: api/
   :template: class.rst

   MeanTransform
   SumTransform
   MedianTransform
   MaxTransform
   MinTransform
   QuantileTransform
   StdTransform
   MADTransform
   MinMaxDifferenceTransform

Scaling transforms:

.. autosummary::
   :toctree: api/
   :template: class.rst

   StandardScalerTransform
   RobustScalerTransform
   MinMaxScalerTransform
   MaxAbsScalerTransform

Functional transforms:

.. autosummary::
   :toctree: api/
   :template: class.rst

   LambdaTransform
   AddConstTransform
   LogTransform
   YeoJohnsonTransform
   BoxCoxTransform
   DifferencingTransform
