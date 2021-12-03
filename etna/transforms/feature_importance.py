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
    """Transform that selects regressors according to tree-based models feature importance."""

    def __init__(self, model: TreeBasedRegressor, top_k: int):
        """
        Init TreeFeatureSelectionTransform.

        Parameters
        ----------
        model:
            model to make selection, it should have feature_importances_ property
            (e.g. all tree-based regressors in sklearn)
        top_k:
            num of regressors to select; if there are not enough regressors, then all will be selected
        """
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError("Parameter top_k should be positive integer")
        super().__init__()
        self.model = model
        self.top_k = top_k

    @staticmethod
    def _get_train(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get train data for model."""
        regressors = TreeFeatureSelectionTransform._get_regressors(df)
        df = TSDataset.to_flatten(df).dropna()
        train_target = df["target"]
        train_data = df[regressors]
        return train_data, train_target

    def _get_regressors_weights(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get weights for regressors based on model feature importances."""
        train_data, train_target = self._get_train(df)
        self.model.fit(train_data, train_target)
        weights_array = self.model.feature_importances_
        weights_dict = {
            column: weights_array[i] for i, column in enumerate(train_data.columns) if column.startswith("regressor_")
        }
        return weights_dict

    @staticmethod
    def _select_top_k_regressors(weights: Dict[str, float], top_k: int) -> List[str]:
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
        if len(self._get_regressors(df)) == 0:
            warnings.warn("It is not possible to select regressors if there aren't any")
            return self
        weights = self._get_regressors_weights(df)
        self.selected_regressors = self._select_top_k_regressors(weights, self.top_k)
        return self
