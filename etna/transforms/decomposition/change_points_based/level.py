from typing import Optional

from ruptures import Binseg

from etna.transforms.decomposition.change_points_based.base import BaseChangePointsModelAdapter
from etna.transforms.decomposition.change_points_based.base import ReversibleChangePointsTransform
from etna.transforms.decomposition.change_points_based.change_points_models.ruptures_based import (
    RupturesChangePointsModel,
)
from etna.transforms.decomposition.change_points_based.detrend import _OneSegmentChangePointsTrendTransform
from etna.transforms.decomposition.change_points_based.per_interval_models import MeanPerIntervalModel
from etna.transforms.decomposition.change_points_based.per_interval_models import StatisticsPerIntervalModel


class _OneSegmentChangePointsLevelTransform(_OneSegmentChangePointsTrendTransform):
    def __init__(
        self,
        in_column: str,
        change_points_model: BaseChangePointsModelAdapter,
        per_interval_model: StatisticsPerIntervalModel,
    ):
        """Init _OneSegmentChangePointsTransform.

        Parameters
        ----------
        in_column:
            name of column to apple transform to
        change_points_model:
            model to get change points from data
        per_interval_model:
            model to process intervals between change points
        """
        super().__init__(
            in_column=in_column, change_points_model=change_points_model, per_interval_model=per_interval_model
        )


class ChangePointsLevelTransform(ReversibleChangePointsTransform):
    """ChangePointsLevelTransform uses :py:class:`ruptures.detection.Binseg` model as a change point detection model.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: str,
        change_points_model: Optional[BaseChangePointsModelAdapter] = None,
        per_interval_model: Optional[StatisticsPerIntervalModel] = None,
    ):
        """Init ChangePointsTrendTransform.

        Parameters
        ----------
        in_column:
            name of column to apply transform to
        change_points_model:
            model to get trend change points
        per_interval_model:
            model to process intervals of segment
        """
        self.in_column = in_column
        self.change_points_model = (
            change_points_model
            if change_points_model is not None
            else RupturesChangePointsModel(
                change_points_model=Binseg(model="l2"),
                n_bkps=5,
            )
        )
        self.per_interval_model = per_interval_model if per_interval_model is not None else MeanPerIntervalModel()
        super().__init__(
            transform=_OneSegmentChangePointsLevelTransform(
                in_column=self.in_column,
                change_points_model=self.change_points_model,
                per_interval_model=self.per_interval_model,
            ),
            required_features=[in_column],
        )
