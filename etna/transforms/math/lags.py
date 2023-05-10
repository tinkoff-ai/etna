from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import pandas as pd

from etna.datasets import TSDataset
from etna.models.utils import determine_num_steps
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
        self._auto = False

        self._freq: Optional[str] = None
        self._created_regressors: Optional[List[str]] = None
        self._exog_shifts: Optional[Dict[str, int]] = None
        self._exog_last_date: Optional[Dict[str, pd.Timestamp]] = None
        self._filter_out_columns = {"target"}

        if isinstance(lag, int):
            if lag <= 0:
                raise ValueError(f"{self.__class__.__name__} works only with positive lags values, {lag} given")
            self.lag = lag

        else:
            if horizon is None:
                raise ValueError("Value of `horizon` should be specified when using `auto`!")

            if horizon < 1:
                raise ValueError(f"{self.__class__.__name__} works only with positive horizon values, {horizon} given")

            self.horizon = horizon
            self._auto = True

    def _save_exog_last_date(self, df_exog: Optional[pd.DataFrame] = None):
        """Save last available date of each exogenous variable."""
        self._exog_last_date = {}
        if df_exog is not None:
            exog_names = set(df_exog.columns.get_level_values("feature"))

            for name in exog_names:
                feature = df_exog.loc[:, pd.IndexSlice[:, name]]

                na_mask = pd.isna(feature).any(axis=1)
                last_date = feature.index[~na_mask].max()

                self._exog_last_date[name] = last_date

    def fit(self, ts: TSDataset) -> "ExogShiftTransform":
        """Fit the transform.

        Parameters
        ----------
        ts:
            Dataset to fit the transform on.

        Returns
        -------
        :
            The fitted transform instance.
        """
        self._freq = ts.freq
        self._save_exog_last_date(df_exog=ts.df_exog)

        super().fit(ts=ts)

        return self

    def _fit(self, df: pd.DataFrame) -> "ExogShiftTransform":
        """Estimate shifts for exogenous variables.

        Parameters
        ----------
        df:
            dataframe with data.

        Returns
        -------
        :
            Fitted `ExogShiftTransform` instance.
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

    def _get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Return the names of exogenous variables."""
        feature_names = []
        if self._exog_last_date is not None:
            feature_names = list(self._exog_last_date.keys())

        df_columns = df.columns.get_level_values("feature")
        for name in feature_names:
            if name not in df_columns:
                raise ValueError(f"Feature `{name}` is expected to be in the dataframe!")

        return feature_names

    def _estimate_shift(self, df: pd.DataFrame, feature_name: str) -> int:
        """Estimate shift value for exogenous variable."""
        if not self._auto:
            return self.lag  # type: ignore

        if self._exog_last_date is None or self._freq is None:
            raise ValueError("Call `fit()` method before estimating exog shifts!")

        last_date = df.index.max()
        last_feature_date = self._exog_last_date[feature_name]

        if last_feature_date > last_date:
            delta = -determine_num_steps(start_timestamp=last_date, end_timestamp=last_feature_date, freq=self._freq)

        elif last_feature_date < last_date:
            delta = determine_num_steps(start_timestamp=last_feature_date, end_timestamp=last_date, freq=self._freq)

        else:
            delta = 0

        shift = max(0, delta + self.horizon)  # type: ignore

        return shift

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shift exogenous variables.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        :
            Transformed dataframe.
        """
        if self._exog_shifts is None:
            raise ValueError("Transform is not fitted!")

        result = df
        segments = sorted(set(df.columns.get_level_values("segment")))
        feature_names = self._get_feature_names(df=df)

        shifted_features = []
        features_to_remove = []
        for feature_name in feature_names:
            shift = self._exog_shifts[feature_name]

            feature = df.loc[:, pd.IndexSlice[:, feature_name]]

            if shift > 0:
                shifted_feature = feature.shift(shift, freq=self._freq)

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
            raise ValueError("Fit the transform to get the regressors info!")

        return self._created_regressors
