from typing import Dict
from typing import List

import pandas as pd

from etna.analysis import get_sequence_anomalies
from etna.datasets import TSDataset
from etna.transforms.outliers.base import OutliersTransform


class SAXOutliersTransform(OutliersTransform):
    """Transform that uses get_sequence_anomalies to find anomalies in data and replaces them with NaN."""

    def __init__(
        self,
        in_column: str,
        num_anomalies: int = 1,
        anomaly_length: int = 15,
        alphabet_size: int = 3,
        word_length: int = 3,
    ):
        """Create instance of SAXOutliersTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        num_anomalies:
            number of outliers to be found
        anomaly_length:
            target length of outliers
        alphabet_size:
            the number of letters with which the subsequence will be encrypted
        word_length:
            the number of segments into which the subsequence will be divided by the paa algorithm
        """
        self.num_anomalies = num_anomalies
        self.anomaly_length = anomaly_length
        self.alphabet_size = alphabet_size
        self.word_length = word_length
        super().__init__(in_column=in_column)

    def detect_outliers(self, ts: TSDataset) -> Dict[str, List[pd.Timestamp]]:
        """Call `get_sequence_anomalies` function with self parameters.

        Parameters
        ----------
        ts:
            dataset to process

        Returns
        -------
        dict of outliers:
            dict of outliers in format {segment: [outliers_timestamps]}
        """
        return get_sequence_anomalies(
            ts=ts,
            in_column=self.in_column,
            num_anomalies=self.num_anomalies,
            anomaly_length=self.anomaly_length,
            alphabet_size=self.alphabet_size,
            word_length=self.word_length,
        )
