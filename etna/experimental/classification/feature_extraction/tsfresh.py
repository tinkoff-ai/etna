from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from tsfresh import extract_features

from etna.experimental.classification.feature_extraction.base import BaseTimeSeriesFeatureExtractor


class TSFreshFeatureExtractor(BaseTimeSeriesFeatureExtractor):
    """Сlass to hold tsfresh features extraction from tsfresh.

    Notes
    -----
    tsfresh should be installed separately using `pip install tsfresh`.
    """

    def __init__(self, default_fc_parameters: dict, n_jobs: int = 1, **kwargs):
        """Init TSFreshFeatureExtractor with given parameters.

        Parameters
        ----------
        default_fc_parametrs:
            Dict with names of features.
            .. Examples: https://github.com/blue-yonder/tsfresh/blob/main/tsfresh/feature_extraction/settings.py
        n_jobs:
            The number of processes to use for parallelization.
        """
        self.default_fc_parameters = default_fc_parameters
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def fit(self, x: List[np.ndarray], y: Optional[np.ndarray] = None) -> "TSFreshFeatureExtractor":
        """Fit the feature extractor."""
        return self

    def transform(self, x: List[np.ndarray]) -> np.ndarray:
        """Extract tsfresh features from the input data.

        Parameters
        ----------
        x:
            Array with time series.

        Returns
        -------
        :
            Transformed input data.
        """
        df_tsfresh = pd.concat([pd.DataFrame({"id": i, "value": series}) for i, series in enumerate(x)])
        df_features = extract_features(
            timeseries_container=df_tsfresh,
            column_id="id",
            column_value="value",
            default_fc_parameters=self.default_fc_parameters,
            n_jobs=self.n_jobs,
            **self.kwargs,
        )
        # TODO: there might be different number of features for train/test set
        df_features.dropna(axis=1, inplace=True)
        return df_features.values
