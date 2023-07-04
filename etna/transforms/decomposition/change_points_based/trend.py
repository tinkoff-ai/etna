from typing import Dict
from typing import Optional

import pandas as pd
from ruptures import Binseg
from sklearn.linear_model import LinearRegression

from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.distributions import IntDistribution
from etna.transforms.decomposition.change_points_based.base import IrreversibleChangePointsTransform
from etna.transforms.decomposition.change_points_based.change_points_models import BaseChangePointsModelAdapter
from etna.transforms.decomposition.change_points_based.change_points_models.ruptures_based import (
    RupturesChangePointsModel,
)
from etna.transforms.decomposition.change_points_based.detrend import _OneSegmentChangePointsTrendTransform
from etna.transforms.decomposition.change_points_based.per_interval_models import PerIntervalModel
from etna.transforms.decomposition.change_points_based.per_interval_models import SklearnRegressionPerIntervalModel


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

    _default_change_points_model = RupturesChangePointsModel(
        change_points_model=Binseg(model="ar"),
        n_bkps=5,
    )
    _default_per_interval_model = SklearnRegressionPerIntervalModel(model=LinearRegression())

    def __init__(
        self,
        in_column: str,
        change_points_model: Optional[BaseChangePointsModelAdapter] = None,
        per_interval_model: Optional[PerIntervalModel] = None,
        out_column: Optional[str] = None,
    ):
        """Init TrendTransform.

        Parameters
        ----------
        in_column:
            name of column to apply transform to
        change_points_model:
            model to get trend change points,
            by default :py:class:`ruptures.detection.Binseg` in a wrapper with ``n_bkps=5`` is used
        per_interval_model:
            model to process intervals of segment,
            by default :py:class:`sklearn.linear_models.LinearRegression` in a wrapper is used
        out_column:
            name of added column.
            If not given, use ``self.__repr__()``
        """
        self.in_column = in_column
        self.out_column = out_column

        self.change_points_model = (
            change_points_model if change_points_model is not None else self._default_change_points_model
        )
        self.per_interval_model = (
            per_interval_model if per_interval_model is not None else self._default_per_interval_model
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

    @property
    def _is_change_points_model_default(self) -> bool:
        # it can't see the difference between Binseg(model="ar") and Binseg(model="l1")
        return self.change_points_model.to_dict() == self._default_change_points_model.to_dict()

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        If ``self.change_points_model`` is equal to default then this grid tunes parameters:
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
                "change_points_model.n_bkps": IntDistribution(low=5, high=30),
            }
        else:
            return {}
