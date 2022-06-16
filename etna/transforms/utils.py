import re
import warnings
from typing import Any
from typing import List
from typing import Set

import pandas as pd

from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform


def match_target_quantiles(features: Set[str]) -> Set[str]:
    """Find quantiles in dataframe columns."""
    pattern = re.compile("target_\d+\.\d+$")
    return {i for i in list(features) if pattern.match(i) is not None}


class _OneSegmentTruncateTransform(Transform):
    """Instance of this class applies truncate transformation to one segment data."""

    def __init__(self, in_column: str, mask_column: str, segments_to_truncate: List[Any]):
        """
        Init OneSegmentTruncateTransform.

        Parameters
        ----------
        in_column:
             column to apply transform.
        mask_column:
               name of mask column
        segments_to_truncate:
               contains values, which should be selected in mask_column
        """
        self.in_column = in_column
        self.mask_column = mask_column
        self.segments_to_truncate = segments_to_truncate

    def fit(self, df: pd.DataFrame) -> "_OneSegmentTruncateTransform":
        """Fit preprocess method, does nothing in OneSegmentLogTransform case.

        Returns
        -------
        :
        """
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply truncated transformation to series from df.

        Parameters
        ----------
        df:
            series to transform

        Returns
        -------
        :
            transformed series

        Warnings
        --------
                throws if some of values from segments_to_truncate are not in values from mask_column
        """
        for segment in self.segments_to_truncate:
            if segment not in df[self.mask_column]:
                warnings.warn(f"Value {segment} not in column {self.mask_column} of dataframe")

        result_df = df.copy()
        mask_set_none = result_df[self.mask_column].isin(self.segments_to_truncate)
        result_df[self.in_column][mask_set_none] = None
        return result_df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply inverse transformation to the series from df.

        Parameters
        ----------
        df:
            series to transform

        Returns
        -------
            :
            transformed series
        """
        return df.copy()


class TruncateTransform(PerSegmentWrapper):
    """TruncateTransform assigns nan in the in_column to those elements whose values in the mask_colum contained in segments_to_truncate."""

    def __init__(self, in_column: str, mask_column: str, segments_to_truncate: List[Any]):
        """Create instance of TruncateTransform.

        Parameters
        ----------
        in_column:
               name of processed column
        mask_column:
               name of mask column
        segments_to_truncate:
               contains values, which should be selected in mask_column
        Warnings
        --------
                throws if some of values from segments_to_truncate are not in values from mask_column
        """
        self.in_column = in_column
        self.mask_column = mask_column
        self.segments_to_truncate = segments_to_truncate
        self.out_column = in_column
        super().__init__(
            transform=_OneSegmentTruncateTransform(
                in_column=in_column, mask_column=mask_column, segments_to_truncate=segments_to_truncate
            )
        )
