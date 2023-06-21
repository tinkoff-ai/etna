from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd
from ruptures import Binseg

from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.distributions import IntDistribution
from etna.transforms.decomposition.change_points_based.base import IrreversibleChangePointsTransform
from etna.transforms.decomposition.change_points_based.base import _OneSegmentChangePointsTransform
from etna.transforms.decomposition.change_points_based.change_points_models import BaseChangePointsModelAdapter
from etna.transforms.decomposition.change_points_based.change_points_models.ruptures_based import (
    RupturesChangePointsModel,
)
from etna.transforms.decomposition.change_points_based.per_interval_models import ConstantPerIntervalModel


class _OneSegmentChangePointsSegmentationTransform(_OneSegmentChangePointsTransform):
    """_OneSegmentChangePointsSegmentationTransform make label encoder to change points."""

    def __init__(self, in_column: str, out_column: str, change_points_model: BaseChangePointsModelAdapter):
        """Init _OneSegmentChangePointsSegmentationTransform.
        Parameters
        ----------
        in_column:
            name of column to apply transform to
        out_column:
            result column name. If not given use ``self.__repr__()``
        change_points_model:
            model to get change points
        """
        self.out_column = out_column
        super().__init__(
            in_column=in_column,
            change_points_model=change_points_model,
            per_interval_model=ConstantPerIntervalModel(),
        )

    def _fit_per_interval_models(self, series: pd.Series):
        """Fit per-interval models with corresponding data from series."""
        if self.intervals is None or self.per_interval_models is None:
            raise ValueError("Something went wrong on fit! Check the parameters of the transform.")
        for k, interval in enumerate(self.intervals):
            self.per_interval_models[interval].fit(features=np.array([]), target=np.array([]), value=k)

    def _apply_transformation(self, df: pd.DataFrame, transformed_series: pd.Series) -> pd.DataFrame:
        df.loc[:, self.out_column] = transformed_series.astype(int).astype("category")
        return df

    def _apply_inverse_transformation(self, df: pd.DataFrame, transformed_series: pd.Series) -> pd.DataFrame:
        return df


class ChangePointsSegmentationTransform(IrreversibleChangePointsTransform):
    """Transform that makes label encoding of change-point intervals.

    Transform divides each segment into intervals using ``change_points_model``.
    Each interval is enumerated based on its index from the start of the segment.
    New column is created with number of interval for each timestamp.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    _default_change_points_model = RupturesChangePointsModel(
        change_points_model=Binseg(model="ar"),
        n_bkps=5,
    )

    def __init__(
        self,
        in_column: str,
        change_points_model: Optional[BaseChangePointsModelAdapter] = None,
        out_column: Optional[str] = None,
    ):
        """Init ChangePointsSegmentationTransform.

        Parameters
        ----------
        in_column:
            name of column to fit change point model
        change_points_model:
            model to get change points,
            by default :py:class:`ruptures.detection.Binseg` in a wrapper with ``n_bkps=5`` is used
        out_column:
            result column name. If not given use ``self.__repr__()``
        """
        self.in_column = in_column
        self.out_column = out_column if out_column is not None else self.__repr__()

        self.change_points_model = (
            change_points_model if change_points_model is not None else self._default_change_points_model
        )

        super().__init__(
            transform=_OneSegmentChangePointsSegmentationTransform(
                in_column=self.in_column,
                change_points_model=self.change_points_model,
                out_column=self.out_column,
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
