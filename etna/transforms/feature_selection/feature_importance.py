import warnings
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from typing_extensions import Literal

from etna.analysis import RelevanceTable
from etna.analysis.feature_selection.mrmr_selection import AggregationMode
from etna.analysis.feature_selection.mrmr_selection import mrmr
from etna.datasets import TSDataset
from etna.transforms.feature_selection import BaseFeatureSelectionTransform

TreeBasedRegressor = Union[
    DecisionTreeRegressor,
    ExtraTreeRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    CatBoostRegressor,
]


class TreeFeatureSelectionTransform(BaseFeatureSelectionTransform):
    """Transform that selects features according to tree-based models feature importance.

    Notes
    -----
    Transform works with any type of features, however most of the models works only with regressors.
    Therefore, it is recommended to pass the regressors into the feature selection transforms.
    """

    def __init__(
        self,
        model: TreeBasedRegressor,
        top_k: int,
        features_to_use: Union[List[str], Literal["all"]] = "all",
        return_features: bool = False,
    ):
        """
        Init TreeFeatureSelectionTransform.

        Parameters
        ----------
        model:
            model to make selection, it should have ``feature_importances_`` property
            (e.g. all tree-based regressors in sklearn)
        top_k:
            num of features to select; if there are not enough features, then all will be selected
        features_to_use:
            columns of the dataset to select from; if "all" value is given, all columns are used
        return_features:
            indicates whether to return features or not.
        """
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError("Parameter top_k should be positive integer")
        super().__init__(features_to_use=features_to_use, return_features=return_features)
        self.model = model
        self.top_k = top_k

    def _get_train(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get train data for model."""
        features = self._get_features_to_use(df)
        df = TSDataset.to_flatten(df).dropna()
        train_target = df["target"]
        train_data = df[features]
        return train_data, train_target

    def _get_features_weights(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get weights for features based on model feature importances."""
        train_data, train_target = self._get_train(df)
        self.model.fit(train_data, train_target)
        weights_array = self.model.feature_importances_
        weights_dict = {column: weights_array[i] for i, column in enumerate(train_data.columns)}
        return weights_dict

    @staticmethod
    def _select_top_k_features(weights: Dict[str, float], top_k: int) -> List[str]:
        keys = np.array(list(weights.keys()))
        values = np.array(list(weights.values()))
        idx_sort = np.argsort(values)[::-1]
        idx_selected = idx_sort[:top_k]
        return keys[idx_selected].tolist()

    def fit(self, df: pd.DataFrame) -> "TreeFeatureSelectionTransform":
        """
        Fit the model and remember features to select.

        Parameters
        ----------
        df:
            dataframe with all segments data

        Returns
        -------
        result: TreeFeatureSelectionTransform
            instance after fitting
        """
        if len(self._get_features_to_use(df)) == 0:
            warnings.warn("It is not possible to select features if there aren't any")
            return self
        weights = self._get_features_weights(df)
        self.selected_features = self._select_top_k_features(weights, self.top_k)
        return self


class MRMRFeatureSelectionTransform(BaseFeatureSelectionTransform):
    """Transform that selects features according to MRMR variable selection method adapted to the timeseries case.

    Notes
    -----
    Transform works with any type of features, however most of the models works only with regressors.
    Therefore, it is recommended to pass the regressors into the feature selection transforms.
    """

    def __init__(
        self,
        relevance_table: RelevanceTable,
        top_k: int,
        features_to_use: Union[List[str], Literal["all"]] = "all",
        relevance_aggregation_mode: str = AggregationMode.mean,
        redundancy_aggregation_mode: str = AggregationMode.mean,
        atol: float = 1e-10,
        return_features: bool = False,
        **relevance_params,
    ):
        """
        Init MRMRFeatureSelectionTransform.

        Parameters
        ----------
        relevance_table:
            method to calculate relevance table
        top_k:
            num of features to select; if there are not enough features, then all will be selected
        features_to_use:
            columns of the dataset to select from
            if "all" value is given, all columns are used
        relevance_aggregation_mode:
            the method for relevance values per-segment aggregation
        redundancy_aggregation_mode:
            the method for redundancy values per-segment aggregation
        atol:
            the absolute tolerance to compare the float values
        return_features:
            indicates whether to return features or not.
        """
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError("Parameter top_k should be positive integer")

        super().__init__(features_to_use=features_to_use, return_features=return_features)
        self.relevance_table = relevance_table
        self.top_k = top_k
        self.relevance_aggregation_mode = relevance_aggregation_mode
        self.redundancy_aggregation_mode = redundancy_aggregation_mode
        self.atol = atol
        self.relevance_params = relevance_params

    def fit(self, df: pd.DataFrame) -> "MRMRFeatureSelectionTransform":
        """
        Fit the method and remember features to select.

        Parameters
        ----------
        df:
            dataframe with all segments data

        Returns
        -------
        result: MRMRFeatureSelectionTransform
            instance after fitting
        """
        features = self._get_features_to_use(df)
        ts = TSDataset(df=df, freq=pd.infer_freq(df.index))
        relevance_table = self.relevance_table(ts[:, :, "target"], ts[:, :, features], **self.relevance_params)
        if not self.relevance_table.greater_is_better:
            relevance_table *= -1
        self.selected_features = mrmr(
            relevance_table=relevance_table,
            regressors=ts[:, :, features],
            top_k=self.top_k,
            relevance_aggregation_mode=self.relevance_aggregation_mode,
            redundancy_aggregation_mode=self.redundancy_aggregation_mode,
            atol=self.atol,
        )
        return self
