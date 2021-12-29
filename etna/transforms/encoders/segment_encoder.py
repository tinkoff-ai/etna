import pandas as pd
from sklearn import preprocessing

from etna.transforms.base import Transform


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
