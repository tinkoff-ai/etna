import warnings
from typing import List
from typing import Optional

import pandas as pd

from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform


class _OneSegmentResampleWithDistributionTransform(Transform):
    """_OneSegmentResampleWithDistributionTransform resamples the given column using the distribution of the other column."""

    def __init__(self, in_column: str, distribution_column: str, inplace: bool, out_column: Optional[str]):
        """
        Init _OneSegmentResampleWithDistributionTransform.

        Parameters
        ----------
        in_column:
            name of column to be resampled
        distribution_column:
            name of column to obtain the distribution from
        inplace:

            * if True, apply resampling inplace to in_column,

            * if False, add transformed column to dataset

        out_column:
            name of added column. If not given, use ``self.__repr__()``
        """
        self.in_column = in_column
        self.distribution_column = distribution_column
        self.inplace = inplace
        self.out_column = out_column
        self.distribution: pd.DataFrame = None

    def _get_folds(self, df: pd.DataFrame) -> List[int]:
        """
        Generate fold number for each timestamp of the dataframe.

        Here the ``in_column`` frequency gap is divided into the folds with the size of dataset frequency gap.
        """
        in_column_index = df[self.in_column].dropna().index
        if len(in_column_index) <= 1 or (len(in_column_index) >= 3 and not pd.infer_freq(in_column_index)):
            raise ValueError(
                "Can not infer in_column frequency!"
                "Check that in_column frequency is compatible with dataset frequency."
            )
        in_column_freq = in_column_index[1] - in_column_index[0]
        dataset_freq = df.index[1] - df.index[0]
        n_folds_per_gap = in_column_freq // dataset_freq
        n_periods = len(df) // n_folds_per_gap + 2

        in_column_start_index = in_column_index[0]
        left_tie_len = len(df[:in_column_start_index]) - 1
        right_tie_len = len(df[in_column_start_index:])
        folds_for_left_tie = list(range(n_folds_per_gap - left_tie_len, n_folds_per_gap))
        folds_for_right_tie = [fold for _ in range(n_periods) for fold in range(n_folds_per_gap)][:right_tie_len]
        return folds_for_left_tie + folds_for_right_tie

    def fit(self, df: pd.DataFrame) -> "_OneSegmentResampleWithDistributionTransform":
        """
        Obtain the resampling frequency and distribution from ``distribution_column``.

        Parameters
        ----------
        df:
            dataframe with data to fit the transform.

        Returns
        -------
        :
        """
        df = df[[self.in_column, self.distribution_column]]
        df["fold"] = self._get_folds(df=df)
        self.distribution = df[["fold", self.distribution_column]].groupby("fold").sum().reset_index()
        self.distribution[self.distribution_column] /= self.distribution[self.distribution_column].sum()
        self.distribution.rename(columns={self.distribution_column: "distribution"}, inplace=True)
        self.distribution.columns.name = None
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample the `in_column` using the distribution of `distribution_column`.

        Parameters
        ----------
        df
            dataframe with data to transform.

        Returns
        -------
        :
            result dataframe
        """
        df["fold"] = self._get_folds(df)
        df = df.reset_index().merge(self.distribution, on="fold").set_index("timestamp").sort_index()
        df[self.out_column] = df[self.in_column].ffill() * df["distribution"]
        df = df.drop(["fold", "distribution"], axis=1)
        return df


class ResampleWithDistributionTransform(PerSegmentWrapper):
    """ResampleWithDistributionTransform resamples the given column using the distribution of the other column.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self, in_column: str, distribution_column: str, inplace: bool = True, out_column: Optional[str] = None
    ):
        """
        Init ResampleWithDistributionTransform.

        Parameters
        ----------
        in_column:
            name of column to be resampled
        distribution_column:
            name of column to obtain the distribution from
        inplace:

            * if True, apply resampling inplace to in_column,

            * if False, add transformed column to dataset

        out_column:
            name of added column. If not given, use ``self.__repr__()``
        """
        self.in_column = in_column
        self.distribution_column = distribution_column
        self.inplace = inplace
        self.out_column = self._get_out_column(out_column)
        super().__init__(
            transform=_OneSegmentResampleWithDistributionTransform(
                in_column=in_column,
                distribution_column=distribution_column,
                inplace=inplace,
                out_column=self.out_column,
            )
        )

    def _get_out_column(self, out_column: Optional[str]) -> str:
        """Get the `out_column` depending on the transform's parameters."""
        if self.inplace and out_column:
            warnings.warn("Transformation will be applied inplace, out_column param will be ignored")
        if self.inplace:
            return self.in_column
        if out_column:
            return out_column
        return self.__repr__()
