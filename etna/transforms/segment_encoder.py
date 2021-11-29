import numpy as np
import pandas as pd
from sklearn import preprocessing

from etna.transforms.base import Transform
from etna.transforms.statistics import MeanTransform


class SegmentEncoderTransform(Transform):
    """Encode segment label to categorical. Creates column 'regressor_segment_code'."""

    idx = pd.IndexSlice

    def __init__(self):
        self._le = preprocessing.LabelEncoder()

    def fit(self, df: pd.DataFrame) -> "SegmentEncoderTransform":
        """
        Fit encoder on existing segment labels.

        Parameters
        ----------
        df:
            dataframe with data to fit label encoder.

        Returns
        -------
        self
        """
        segment_columns = df.columns.get_level_values("segment")
        self._le.fit(segment_columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get encoded (categorical) for each segment.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result dataframe
        """
        encoded_matrix = self._le.transform(self._le.classes_)
        encoded_matrix = encoded_matrix.reshape(len(self._le.classes_), -1).repeat(len(df), axis=1).T
        encoded_df = pd.DataFrame(
            encoded_matrix,
            columns=pd.MultiIndex.from_product(
                [self._le.classes_, ["regressor_segment_code"]], names=("segment", "feature")
            ),
            index=df.index,
        )
        encoded_df = encoded_df.astype("category")
        df = df.join(encoded_df)
        df = df.sort_index(axis=1)
        return df


class MeanSegmentEncoderTransform(Transform):
    """Makes expanding mean target encoding of the segment. Creates column 'regressor_segment_mean'."""

    idx = pd.IndexSlice

    def __init__(self):
        self.mean_encoder = MeanTransform(in_column="target", window=-1, out_column="regressor_segment_mean")
        self.global_means: np.ndarray[float] = None

    def fit(self, df: pd.DataFrame) -> "MeanSegmentEncoderTransform":
        """
        Fit encoder.

        Parameters
        ----------
        df:
            dataframe with data to fit expanding mean target encoder.

        Returns
        -------
        self
        """
        self.mean_encoder.fit(df)
        self.global_means = df.loc[:, self.idx[:, "target"]].mean().values
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get encoded values for the segment.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result dataframe
        """
        df = self.mean_encoder.transform(df)
        segment = df.columns.get_level_values("segment").unique()[0]
        nan_timestamps = df[df.loc[:, self.idx[segment, "target"]].isna()].index
        df.loc[nan_timestamps, self.idx[:, "regressor_segment_mean"]] = self.global_means
        return df
