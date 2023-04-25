from typing import Dict
from typing import Optional

import pandas as pd
from ruptures import Binseg

from etna import SETTINGS
from etna.transforms.decomposition.change_points_based.base import IrreversibleChangePointsTransform
from etna.transforms.decomposition.change_points_based.change_points_models import BaseChangePointsModelAdapter
from etna.transforms.decomposition.change_points_based.change_points_models.ruptures_based import (
    RupturesChangePointsModel,
)
from etna.transforms.decomposition.change_points_based.detrend import _OneSegmentChangePointsTrendTransform
from etna.transforms.decomposition.change_points_based.per_interval_models import PerIntervalModel
from etna.transforms.decomposition.change_points_based.per_interval_models import SklearnRegressionPerIntervalModel

if SETTINGS.auto_required:
    from optuna.distributions import BaseDistribution
    from optuna.distributions import CategoricalDistribution
    from optuna.distributions import IntUniformDistribution


class _OneSegmentTrendTransform(_OneSegmentChangePointsTrendTransform):
    """_OneSegmentTrendTransform adds trend as a feature."""

    def __init__(
        self,
        in_column: str,
        out_column: str,
        change_points_model: BaseChangePointsModelAdapter,
        per_interval_model: PerIntervalModel,
    ):
        """Init _OneSegmentTrendTransform.

        Parameters
        ----------
        in_column:
            name of column to apply transform to
        out_column:
            name of added column
        change_points_model:
            model to get trend change points
        per_interval_model:
            model to get trend from data
        """
        self.out_column = out_column
        super().__init__(
            in_column=in_column,
            change_points_model=change_points_model,
            per_interval_model=per_interval_model,
        )

    def _apply_transformation(self, df: pd.DataFrame, transformed_series: pd.Series) -> pd.DataFrame:
        df.loc[:, self.out_column] = transformed_series
        return df

    def _apply_inverse_transformation(self, df: pd.DataFrame, transformed_series: pd.Series) -> pd.DataFrame:
        return df


class TrendTransform(IrreversibleChangePointsTransform):
    """Transform that adds trend as a feature.

    Transform divides each segment into intervals using ``change_points_model``.
    Then a separate model is fitted on each interval using ``per_interval_model``.
    New column is created with values predicted by the model of each interval.

    Evaluated function can be linear, mean, median, etc. Look at the signature to find out which models can be used.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: str,
        change_points_model: BaseChangePointsModelAdapter = None,
        per_interval_model: Optional[PerIntervalModel] = None,
        out_column: Optional[str] = None,
    ):
        """Init TrendTransform.

        Parameters
        ----------
        in_column:
            name of column to apply transform to
        out_column:
            name of added column.
            If not given, use ``self.__repr__()``
        """
        self.in_column = in_column
        self.out_column = out_column

        self._is_change_points_model_default = change_points_model is None
        self.change_points_model = (
            change_points_model
            if change_points_model is not None
            else RupturesChangePointsModel(
                change_points_model=Binseg(model="l2"),
                n_bkps=5,
            )
        )
        self.per_interval_model = (
            SklearnRegressionPerIntervalModel() if per_interval_model is None else per_interval_model
        )

        super().__init__(
            transform=_OneSegmentTrendTransform(
                in_column=self.in_column,
                out_column=self.out_column if self.out_column is not None else f"{self.__repr__()}",
                change_points_model=self.change_points_model,
                per_interval_model=self.per_interval_model,
            ),
            required_features=[in_column],
        )

    def params_to_tune(self) -> Dict[str, "BaseDistribution"]:
        """Get default grid for tuning hyperparameters.

        If ``change_points_model`` isn't set then this grid tunes parameters:
        ``change_points_model.change_points_model.model``, ``change_points_model.n_bkps``.
        Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        if self._is_change_points_model_default:
            return {
                "change_points_model.change_points_model.model": CategoricalDistribution(
                    ["l1", "l2", "normal", "rbf", "cosine", "linear", "clinear", "ar", "mahalanobis", "rank"]
                ),
                "change_points_model.n_bkps": IntUniformDistribution(low=5, high=30),
            }
        else:
            return {}
