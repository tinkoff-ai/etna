from typing import List

import pandas as pd
from sklearn import preprocessing

from etna.transforms.base import FutureMixin
from etna.transforms.base import IrreversibleTransform


class SegmentEncoderTransform(IrreversibleTransform, FutureMixin):
    """Encode segment label to categorical. Creates column 'segment_code'."""

    idx = pd.IndexSlice

    def __init__(self):
        super().__init__(required_features=["target"])
        self._le = preprocessing.LabelEncoder()

    def _fit(self, df: pd.DataFrame) -> "SegmentEncoderTransform":
        """
        Fit encoder on existing segment labels.

        Parameters
        ----------
        df:
            dataframe with data to fit label encoder.
        Returns
        -------
        :
            Fitted transform
        """
        segment_columns = df.columns.get_level_values("segment")
        self._le.fit(segment_columns)
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get encoded (categorical) for each segment.
        Parameters
        ----------
        df:
            dataframe with data to transform.
        Returns
        -------
        :
            result dataframe
        """
        encoded_matrix = self._le.transform(self._le.classes_)
        encoded_matrix = encoded_matrix.reshape(len(self._le.classes_), -1).repeat(len(df), axis=1).T
        encoded_df = pd.DataFrame(
            encoded_matrix,
            columns=pd.MultiIndex.from_product([self._le.classes_, ["segment_code"]], names=("segment", "feature")),
            index=df.index,
        )
        encoded_df = encoded_df.astype("category")
        df = df.join(encoded_df)
        df = df.sort_index(axis=1)
        return df

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform.
        Returns
        -------
        :
            List with regressors created by the transform.
        """
        return ["segment_code"]
