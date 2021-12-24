from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd

from etna.transforms.base import Transform


# TODO: do smth with quantiles
# TODO: to understand what to do with regressors columns
# TODO: consider adding some checks on the nans inside series
class _SingleDifferencingTransform(Transform):
    """Calculate a time series difference of order 1."""

    def __init__(
        self,
        in_column: str,
        period: int = 1,
        inplace: bool = True,
        out_column: Optional[str] = None,
    ):
        """Create instance of _SingleDifferencingTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        period:
            number of steps back to calculate the difference with, it should be >= 1
        inplace:
            if True, apply transformation inplace to in_column, if False, add transformed column to dataset
        out_column:
            if set, name of added column, the final name will be '{out_column}',
            don't forget to add 'regressor_' prefix
            if don't set, name will be '{repr}'

        Raises
        ------
        ValueError:
            if period is not integer >= 1

        Notes
        -----
        To understand how transform works we recommend: https://otexts.com/fpp2/stationarity.html
        """
        self.in_column = in_column

        if not isinstance(period, int) or period < 1:
            raise ValueError("Period should be at least 2")
        self.period = period

        self.inplace = inplace
        self.out_column = out_column

        self._train_timestamp: Optional[pd.DatetimeIndex] = None
        self._train_init_dict: Optional[Dict[str, pd.Series]] = None
        self._test_init_df: Optional[pd.DataFrame] = None

    def _get_column_name(self) -> str:
        if self.out_column is None:
            return self.__repr__()
        else:
            return self.out_column

    def fit(self, df: pd.DataFrame) -> "_SingleDifferencingTransform":
        """Fit method does nothing and is kept for compatibility.

        Parameters
        ----------
        df:
            dataframe with data.

        Returns
        -------
        result: _SingleDifferencingTransform
        """
        segments = sorted(set(df.columns.get_level_values("segment")))
        fit_df = df.loc[:, pd.IndexSlice[segments, self.in_column]].copy()

        self._train_timestamp = fit_df.index
        self._train_init_dict = {}
        for segment in segments:
            cur_series = fit_df.loc[:, pd.IndexSlice[segment, self.in_column]]
            cur_series = cur_series.loc[cur_series.first_valid_index() :]
            self._train_init_dict[segment] = cur_series[: self.period]
        self._test_init_df = fit_df.iloc[-self.period :, :]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make a differencing transformation.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result: pd.Dataframe
            transformed dataframe
        """
        if self._train_init_dict is None or self._test_init_df is None or self._train_timestamp is None:
            raise ValueError("Transform is not fitted")

        segments = sorted(set(df.columns.get_level_values("segment")))
        transformed = df.loc[:, pd.IndexSlice[segments, self.in_column]].copy()
        for segment in segments:
            start_idx = transformed.loc[:, pd.IndexSlice[segment, self.in_column]].first_valid_index()
            transformed.loc[start_idx:, pd.IndexSlice[segment, self.in_column]] = transformed.loc[
                start_idx:, pd.IndexSlice[segment, self.in_column]
            ].diff(periods=self.period)

        if self.inplace:
            result_df = df.copy()
            result_df.loc[:, pd.IndexSlice[segments, self.in_column]] = transformed
        else:
            transformed_features = pd.DataFrame(
                transformed, columns=df.loc[:, pd.IndexSlice[segments, self.in_column]].columns, index=df.index
            )
            column_name = self._get_column_name()
            transformed_features.columns = pd.MultiIndex.from_product([segments, [column_name]])
            result_df = pd.concat((df, transformed_features), axis=1)
            result_df = result_df.sort_index(axis=1)

        return result_df

    def _reconstruct_train(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct the train in inverse_transform."""
        segments = sorted(set(df.columns.get_level_values("segment")))
        result_df = df.copy()

        # impute values for reconstruction and run reconstruction
        for segment in segments:
            init_segment = self._train_init_dict[segment]  # type: ignore
            cur_series = result_df.loc[:, pd.IndexSlice[segment, self.in_column]]
            cur_series[init_segment.index] = init_segment
            for i in range(self.period):
                cur_series.iloc[i :: self.period] = cur_series.iloc[i :: self.period].cumsum()

            result_df.loc[cur_series.index, pd.IndexSlice[segment, self.in_column]] = cur_series
        return result_df

    def _reconstruct_test(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct the test in inverse_transform."""
        segments = sorted(set(df.columns.get_level_values("segment")))
        result_df = df.copy()

        # check that test is right after the train
        expected_min_test_timestamp = pd.date_range(
            start=self._test_init_df.index.max(),  # type: ignore
            periods=2,
            freq=pd.infer_freq(self._train_timestamp),
            closed="right",
        )[0]
        if expected_min_test_timestamp != df.index.min():
            raise ValueError("Test should go after the train without gaps")

        # we can reconstruct the values by concatenating saved fit values before test values
        to_transform = df.loc[:, pd.IndexSlice[segments, self.in_column]].copy()
        to_transform = pd.concat([self._test_init_df, to_transform])

        # run reconstruction
        for i in range(self.period):
            to_transform.iloc[i :: self.period] = to_transform.iloc[i :: self.period].cumsum()

        # save result
        result_df.loc[:, pd.IndexSlice[segments, self.in_column]] = to_transform

        return result_df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply inverse transformation to DataFrame.

        Parameters
        ----------
        df:
            DataFrame to apply inverse transform.

        Returns
        -------
        result: pd.DataFrame
            transformed DataFrame.
        """
        if self._train_init_dict is None or self._test_init_df is None or self._train_timestamp is None:
            raise ValueError("Transform is not fitted")

        if not self.inplace:
            return df

        # determine if we are working with train or test
        if self._train_timestamp.shape[0] == df.index.shape[0] and np.all(self._train_timestamp == df.index):
            # we are on the train
            result_df = self._reconstruct_train(df)

        elif df.index.min() > self._train_timestamp.max():
            # we are on the test
            result_df = self._reconstruct_test(df)

        else:
            raise ValueError("Inverse transform can be applied only to full train or test that should be in the future")

        return result_df
