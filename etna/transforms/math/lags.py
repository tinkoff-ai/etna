from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import pandas as pd

from etna.transforms.base import FutureMixin
from etna.transforms.base import IrreversibleTransform


class LagTransform(IrreversibleTransform, FutureMixin):
    """Generates series of lags from given dataframe."""

    def __init__(self, in_column: str, lags: Union[List[int], int], out_column: Optional[str] = None):
        """Create instance of LagTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        lags:
            int value or list of values for lags computation; if int, generate range of lags from 1 to given value
        out_column:
            base for the name of created columns;

            * if set the final name is '{out_column}_{lag_number}';

            * if don't set, name will be ``transform.__repr__()``,
              repr will be made for transform that creates exactly this column

        Raises
        ------
        ValueError:
            if lags value contains non-positive values
        """
        super().__init__(required_features=[in_column])
        if isinstance(lags, int):
            if lags < 1:
                raise ValueError(f"{type(self).__name__} works only with positive lags values, {lags} given")
            self.lags = list(range(1, lags + 1))
        else:
            if any(lag_value < 1 for lag_value in lags):
                raise ValueError(f"{type(self).__name__} works only with positive lags values")
            self.lags = lags

        self.in_column = in_column
        self.out_column = out_column

    def _get_column_name(self, lag: int) -> str:
        if self.out_column is None:
            temp_transform = LagTransform(in_column=self.in_column, out_column=self.out_column, lags=[lag])
            return repr(temp_transform)
        else:
            return f"{self.out_column}_{lag}"

    def _fit(self, df: pd.DataFrame) -> "LagTransform":
        """Fit method does nothing and is kept for compatibility.

        Parameters
        ----------
        df:
            dataframe with data.

        Returns
        -------
        result: LagTransform
        """
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lags to the dataset.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result: pd.Dataframe
            transformed dataframe
        """
        result = df
        segments = sorted(set(df.columns.get_level_values("segment")))
        all_transformed_features = []
        features = df.loc[:, pd.IndexSlice[:, self.in_column]]
        for lag in self.lags:
            column_name = self._get_column_name(lag)
            transformed_features = features.shift(lag)
            transformed_features.columns = pd.MultiIndex.from_product([segments, [column_name]])
            all_transformed_features.append(transformed_features)
        result = pd.concat([result] + all_transformed_features, axis=1)
        result = result.sort_index(axis=1)
        return result

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return [self._get_column_name(lag) for lag in self.lags]


class ExogShiftTransform(IrreversibleTransform, FutureMixin):
    """Shifts exogenous variables from a given dataframe."""

    def __init__(self, lag: Union[int, Literal["auto"]], horizon: Optional[int] = None):
        """Create instance of ExogShiftTransform.

        Parameters
        ----------
        lag:
            value for shift estimation

            * if set to `int` all exogenous variables will be shifted `lag` steps forward;

            * if set to `auto` minimal shift will be estimated for each variable based on
              the prediction horizon and available timeline

        horizon:
            prediction horizon. Mandatory when set to `lag="auto"`, ignored otherwise
        """
        super().__init__(required_features="all")

        self.lag: Optional[int] = None
        self.horizon: Optional[int] = None
        self.auto = False

        self._created_regressors: Optional[List[str]] = None
        self._exog_shifts: Optional[Dict[str, int]] = None

        if isinstance(lag, int):
            if lag <= 0:
                raise ValueError(f"{type(self).__name__} works only with positive lags values, {lag} given")
            self.lag = lag

        else:
            if horizon is None:
                raise ValueError("`horizon` should be specified when using `auto`!")

            if horizon < 1:
                raise ValueError(f"{type(self).__name__} works only with positive horizon values, {horizon} given")

            self.horizon = horizon
            self.auto = True

    def _fit(self, df: pd.DataFrame) -> "ExogShiftTransform":
        """Estimate shifts for exogenous variables.

        Parameters
        ----------
        df:
            dataframe with data.

        Returns
        -------
        result: ExogShiftTransform
        """
        feature_names = self._get_feature_names(df=df)

        self._exog_shifts = dict()
        self._created_regressors = []

        for feature_name in feature_names:
            shift = self._estimate_shift(df=df, feature_name=feature_name)
            self._exog_shifts[feature_name] = shift

            if shift > 0:
                self._created_regressors.append(f"{feature_name}_shift_{shift}")

        return self

    @staticmethod
    def _get_feature_names(df: pd.DataFrame) -> List[str]:
        """Return the names of exogenous variables."""
        names = set(df.columns.get_level_values("feature"))

        names_to_remove = {"target"}
        names_to_remove |= match_target_quantiles(names)
        names_to_remove |= match_target_components(names)

        features = names - names_to_remove
        return list(features)

    def _estimate_shift(self, df: pd.DataFrame, feature_name: str) -> int:
        """Estimate shift value for exogenous variable."""
        if not self.auto:
            return self.lag  # type: ignore

        freq = pd.infer_freq(df.index)

        last_date = df.index.max() + pd.Timedelta(self.horizon, unit=freq)
        last_feature_date = self._exog_last_date[feature_name]  # type: ignore

        delta = last_date - last_feature_date
        shift = max(0, int(delta / pd.Timedelta(1, unit=freq)))

        return shift

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shift exogenous variables.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result: pd.Dataframe
            transformed dataframe
        """
        if self._exog_shifts is None:
            raise ValueError("Transform is not fitted!")

        result = df
        freq = pd.infer_freq(df.index)
        segments = sorted(set(df.columns.get_level_values("segment")))
        feature_names = self._get_feature_names(df=df)

        shifted_features = []
        features_to_remove = []
        for feature_name in feature_names:
            shift = self._exog_shifts[feature_name]

            feature = df.loc[:, pd.IndexSlice[:, feature_name]]

            if shift > 0:
                shifted_feature = feature.shift(shift, freq=freq)

                column_name = f"{feature_name}_shift_{shift}"
                shifted_feature.columns = pd.MultiIndex.from_product([segments, [column_name]])

                shifted_features.append(shifted_feature)
                features_to_remove.append(feature_name)

        if len(features_to_remove) > 0:
            result = result.drop(columns=pd.MultiIndex.from_product([segments, features_to_remove]))

        result = pd.concat([result] + shifted_features, axis=1)
        result.sort_index(axis=1, inplace=True)
        return result

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        if self._created_regressors is None:
            return []

        return self._created_regressors
