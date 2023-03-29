import reprlib
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

from etna.transforms import IrreversibleTransform
from etna.transforms.base import FutureMixin
from etna.transforms.math.statistics import MeanTransform


class MeanSegmentEncoderTransform(IrreversibleTransform, FutureMixin):
    """Makes expanding mean target encoding of the segment. Creates column 'segment_mean'."""

    idx = pd.IndexSlice

    def __init__(self):
        super().__init__(required_features=["target"])
        self.mean_encoder = MeanTransform(in_column="target", window=-1, out_column="segment_mean")
        self.global_means: Optional[Dict[str, float]] = None

    def _fit(self, df: pd.DataFrame) -> "MeanSegmentEncoderTransform":
        """
        Fit encoder.

        Parameters
        ----------
        df:
            dataframe with data to fit expanding mean target encoder.

        Returns
        -------
        :
            Fitted transform
        """
        self.mean_encoder._fit(df)
        mean_values = df.loc[:, self.idx[:, "target"]].mean().to_dict()
        mean_values = {key[0]: value for key, value in mean_values.items()}
        self.global_means = mean_values
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get encoded values for the segment.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        :
            result dataframe

        Raises
        ------
        ValueError:
            If transform isn't fitted.
        NotImplementedError:
            If there are segments that weren't present during training.
        """
        if self.global_means is None:
            raise ValueError("The transform isn't fitted!")

        segments = df.columns.get_level_values("segment").unique().tolist()
        new_segments = set(segments) - self.global_means.keys()
        if len(new_segments) > 0:
            raise NotImplementedError(
                f"This transform can't process segments that weren't present on train data: {reprlib.repr(new_segments)}"
            )

        df = self.mean_encoder._transform(df)
        segment = segments[0]
        nan_timestamps = df[df.loc[:, self.idx[segment, "target"]].isna()].index
        values_to_set = np.array([self.global_means[x] for x in segments])
        # repetition isn't necessary for pandas >= 1.2
        values_to_set = np.repeat(values_to_set[np.newaxis, :], len(nan_timestamps), axis=0)
        df.loc[nan_timestamps, self.idx[:, "segment_mean"]] = values_to_set
        return df

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return ["segment_mean"]
