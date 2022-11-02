from typing import Optional

import pandas as pd

from etna.transforms.decomposition.change_points_based.base import IrreversibleChangePointsTransform
from etna.transforms.decomposition.change_points_based.base import OneSegmentChangePointsTransform
from etna.transforms.decomposition.change_points_based.change_points_models import BaseChangePointsModelAdapter
from etna.transforms.decomposition.change_points_based.per_interval_models import ConstantPerIntervalModel


class _OneSegmentChangePointsSegmentationTransform(OneSegmentChangePointsTransform):
    """_OneSegmentChangePointsSegmentationTransform make label encoder to change points."""

    def __init__(self, in_column: str, out_column: str, change_point_model: BaseChangePointsModelAdapter):
        """Init _OneSegmentChangePointsSegmentationTransform.
        Parameters
        ----------
        in_column:
            name of column to apply transform to
        out_column:
            result column name. If not given use ``self.__repr__()``
        change_point_model:
            model to get change points
        """
        self.out_column = out_column
        super().__init__(
            in_column=in_column,
            change_point_model=change_point_model,
            per_interval_model=ConstantPerIntervalModel(),
        )

    def _fit_per_interval_models(self, series: pd.Series):
        """Fit per-interval models with corresponding data from series."""
        if self.intervals is None or self.per_interval_models is None:
            raise ValueError("Something went wrong on fit! Check the parameters of the transform.")
        for k, interval in enumerate(self.intervals):
            self.per_interval_models[interval].fit(value=k)

    def _apply_transformation(self, df: pd.DataFrame, transformed_series: pd.Series) -> pd.DataFrame:
        df.loc[:, self.out_column] = transformed_series.astype(int).astype("category")
        return df

    def _apply_inverse_transformation(self, df: pd.DataFrame, transformed_series: pd.Series) -> pd.DataFrame:
        return df


class ChangePointsSegmentationTransform(IrreversibleChangePointsTransform):
    """ChangePointsSegmentationTransform make label encoder to change points.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: str,
        change_point_model: BaseChangePointsModelAdapter,
        out_column: Optional[str] = None,
    ):
        """Init ChangePointsSegmentationTransform.

        Parameters
        ----------
        in_column:
            name of column to fit change point model
        out_column:
            result column name. If not given use ``self.__repr__()``
        change_point_model:
            model to get change points
        """
        self.in_column = in_column
        self.out_column = out_column if out_column is not None else self.__repr__()
        self.change_point_model = change_point_model
        super().__init__(
            transform=_OneSegmentChangePointsSegmentationTransform(
                in_column=self.in_column,
                out_column=self.out_column,
                change_point_model=self.change_point_model,
            ),
            required_features=[in_column],
        )
