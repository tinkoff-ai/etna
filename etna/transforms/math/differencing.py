from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Union
from typing import cast

import numpy as np
import pandas as pd

from etna.datasets import TSDataset
from etna.distributions import BaseDistribution
from etna.distributions import IntDistribution
from etna.transforms.base import ReversibleTransform
from etna.transforms.utils import check_new_segments
from etna.transforms.utils import match_target_quantiles


class _SingleDifferencingTransform(ReversibleTransform):
    """Calculate a time series differences of order 1.

    During ``fit`` this transform can work with NaNs at the beginning of the segment, but fails when meets NaN inside the segment.
    During ``transform`` and ``inverse_transform`` there is no special treatment of NaNs.

    Notes
    -----
    To understand how transform works we recommend:
    `Stationarity and Differencing <https://otexts.com/fpp2/stationarity.html>`_
    """

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

            * if True, apply transformation inplace to in_column,

            * if False, add transformed column to dataset

        out_column:

            * if set, name of added column, the final name will be '{out_column}';

            * if isn't set, name will be based on ``self.__repr__()``

        Raises
        ------
        ValueError:
            if period is not integer >= 1
        """
        super().__init__(required_features=[in_column])
        self.in_column = in_column

        if not isinstance(period, int) or period < 1:
            raise ValueError("Period should be at least 1")
        self.period = period

        self.inplace = inplace
        self.out_column = out_column

        self._train_timestamp: Optional[pd.DatetimeIndex] = None
        self._train_init_dict: Optional[Dict[str, pd.Series]] = None
        self._test_init_df: Optional[pd.DataFrame] = None
        self.in_column_regressor: Optional[bool] = None

    def _get_column_name(self) -> str:
        if self.inplace:
            return self.in_column
        if self.out_column is None:
            return self.__repr__()
        else:
            return self.out_column

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        if self.in_column_regressor is None:
            raise ValueError("Fit the transform to get the correct regressors info!")
        if self.inplace:
            return []
        return [self._get_column_name()] if self.in_column_regressor else []

    def fit(self, ts: TSDataset) -> "_SingleDifferencingTransform":
        """Fit the transform."""
        self.in_column_regressor = self.in_column in ts.regressors
        super().fit(ts)
        return self

    def _fit(self, df: pd.DataFrame) -> "_SingleDifferencingTransform":
        """Fit the transform.

        Parameters
        ----------
        df:
            dataframe with data.

        Returns
        -------
        result: _SingleDifferencingTransform

        Raises
        ------
        ValueError:
            if NaNs are present inside the segment
        """
        segments = sorted(set(df.columns.get_level_values("segment")))
        fit_df = df.loc[:, pd.IndexSlice[segments, self.in_column]].copy()

        train_init_dict = {}
        for current_segment in segments:
            cur_series = fit_df.loc[:, pd.IndexSlice[current_segment, self.in_column]]
            cur_series = cur_series.loc[cur_series.first_valid_index() :]

            if cur_series.isna().sum() > 0:
                raise ValueError(f"There should be no NaNs inside the segments")

            train_init_dict[current_segment] = cur_series[: self.period]

        self._train_init_dict = train_init_dict
        self._train_timestamp = fit_df.index
        self._test_init_df = fit_df.iloc[-self.period :, :]
        # make multiindex levels consistent
        self._test_init_df.columns = self._test_init_df.columns.remove_unused_levels()
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make a differencing transformation.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result:
            transformed dataframe
        """
        segments = sorted(set(df.columns.get_level_values("segment")))
        transformed = df.loc[:, pd.IndexSlice[segments, self.in_column]]
        for current_segment in segments:
            to_transform = transformed.loc[:, pd.IndexSlice[current_segment, self.in_column]]
            # make a differentiation
            transformed.loc[:, pd.IndexSlice[current_segment, self.in_column]] = to_transform.diff(periods=self.period)

        if self.inplace:
            result_df = df
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

    def _make_inv_diff(self, to_transform: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
        """Make inverse difference transform."""
        for i in range(self.period):
            to_transform.iloc[i :: self.period] = to_transform.iloc[i :: self.period].cumsum()
        return to_transform

    def _reconstruct_train(self, df: pd.DataFrame, columns_to_inverse: Set[str]) -> pd.DataFrame:
        """Reconstruct the train in ``inverse_transform``."""
        segments = sorted(set(df.columns.get_level_values("segment")))
        result_df = df.copy()

        # impute values for reconstruction and run reconstruction
        for current_segment in segments:
            init_segment = self._train_init_dict[current_segment]  # type: ignore
            for column in columns_to_inverse:
                cur_series = result_df.loc[:, pd.IndexSlice[current_segment, column]]
                cur_series[init_segment.index] = init_segment.values
                cur_series = self._make_inv_diff(cur_series)
                result_df.loc[:, pd.IndexSlice[current_segment, column]] = cur_series
        return result_df

    def _reconstruct_test(self, df: pd.DataFrame, columns_to_inverse: Set[str]) -> pd.DataFrame:
        """Reconstruct the test in ``inverse_transform``."""
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
        for column in columns_to_inverse:
            to_transform = df.loc[:, pd.IndexSlice[segments, column]].copy()
            init_df = self._test_init_df.copy()  # type: ignore
            init_df.columns.set_levels([column], level="feature", inplace=True)
            init_df = init_df[segments]
            to_transform = pd.concat([init_df, to_transform])

            # run reconstruction and save the result
            to_transform = self._make_inv_diff(to_transform)
            result_df.loc[:, pd.IndexSlice[segments, column]] = to_transform.loc[result_df.index]

        return result_df

    def _fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform dataframe.

        Parameters
        ----------
        df:
            Dataframe to transform.

        Returns
        -------
        :
            Transformed dataframe.
        """
        return self._fit(df=df)._transform(df=df)

    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply inverse transformation to DataFrame.

        Parameters
        ----------
        df:
            DataFrame to apply inverse transform.

        Returns
        -------
        result:
            transformed DataFrame.

        Raises
        ------
        ValueError:
            if inverse transform is applied not to full train nor to test that goes after train
        ValueError:
            if inverse transform is applied to test that goes after train with gap
        """
        # we assume this to be fitted
        self._train_timestamp = cast(pd.DatetimeIndex, self._train_timestamp)

        if not self.inplace:
            return df

        columns_to_inverse = {self.in_column}

        # if we are working with in_column="target" then there can be quantiles to inverse too
        if self.in_column == "target":
            columns_to_inverse.update(match_target_quantiles(set(df.columns.get_level_values("feature"))))

        # determine if we are working with train or test
        if self._train_timestamp.shape[0] == df.index.shape[0] and np.all(self._train_timestamp == df.index):
            # we are on the train
            result_df = self._reconstruct_train(df, columns_to_inverse)

        elif df.index.min() > self._train_timestamp.max():
            # we are on the test
            result_df = self._reconstruct_test(df, columns_to_inverse)

        else:
            raise ValueError("Inverse transform can be applied only to full train or test that should be in the future")

        return result_df


class DifferencingTransform(ReversibleTransform):
    """Calculate a time series differences.

    During ``fit`` this transform can work with NaNs at the beginning of the segment, but fails when meets NaN inside the segment.
    During ``transform`` and ``inverse_transform`` there is no special treatment of NaNs.

    Notes
    -----
    To understand how transform works we recommend:
    `Stationarity and Differencing <https://otexts.com/fpp2/stationarity.html>`_
    """

    def __init__(
        self,
        in_column: str,
        period: int = 1,
        order: int = 1,
        inplace: bool = True,
        out_column: Optional[str] = None,
    ):
        """Create instance of DifferencingTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        period:
            number of steps back to calculate the difference with, it should be >= 1
        order:
            number of differences to make, it should be >= 1
        inplace:

            * if True, apply transformation inplace to in_column,

            * if False, add transformed column to dataset

        out_column:

            * if set, name of added column, the final name will be '{out_column}';

            * if isn't set, name will be based on ``self.__repr__()``

        Raises
        ------
        ValueError:
            if period is not integer >= 1
        ValueError:
            if order is not integer >= 1
        """
        super().__init__(required_features=[in_column])
        self.in_column = in_column

        if not isinstance(period, int) or period < 1:
            raise ValueError("Period should be at least 1")
        self.period = period

        if not isinstance(order, int) or order < 1:
            raise ValueError("Order should be at least 1")
        self.order = order

        self.inplace = inplace
        self.out_column = out_column

        # add differencing transforms for each order
        result_out_column = self._get_column_name()
        self._differencing_transforms: List[_SingleDifferencingTransform] = []
        # first transform should work like this transform but with prepared out_column name
        self._differencing_transforms.append(
            _SingleDifferencingTransform(
                in_column=self.in_column, period=self.period, inplace=self.inplace, out_column=result_out_column
            )
        )
        # other transforms should make differences inplace
        for _ in range(self.order - 1):
            self._differencing_transforms.append(
                _SingleDifferencingTransform(in_column=result_out_column, period=self.period, inplace=True)
            )
        self._fit_segments: Optional[List[str]] = None
        self.in_column_regressor: Optional[bool] = None

    def _get_column_name(self) -> str:
        if self.inplace:
            return self.in_column
        if self.out_column is None:
            return self.__repr__()
        else:
            return self.out_column

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        if self.in_column_regressor is None:
            raise ValueError("Fit the transform to get the correct regressors info!")
        if self.inplace:
            return []
        return [self._get_column_name()] if self.in_column_regressor else []

    def fit(self, ts: TSDataset) -> "DifferencingTransform":
        """Fit the transform."""
        self.in_column_regressor = self.in_column in ts.regressors
        super().fit(ts)
        return self

    def _fit(self, df: pd.DataFrame) -> "DifferencingTransform":
        """Fit the transform.

        Parameters
        ----------
        df:
            dataframe with data.

        Returns
        -------
        result: DifferencingTransform

        Raises
        ------
        ValueError:
            if NaNs are present inside the segment
        """
        # this is made because transforms of high order may need some columns created by transforms of lower order
        result_df = df
        for transform in self._differencing_transforms:
            result_df = transform._fit_transform(result_df)
        self._fit_segments = df.columns.get_level_values("segment").unique().tolist()
        return self

    def _check_is_fitted(self):
        if self._fit_segments is None:
            raise ValueError("Transform is not fitted!")

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make a differencing transformation.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result:
            transformed dataframe

        Raises
        ------
        ValueError:
            if transform isn't fitted
        NotImplementedError:
            if there are segments that weren't present during training
        """
        self._check_is_fitted()
        segments = df.columns.get_level_values("segment").unique().tolist()
        if self.inplace:
            check_new_segments(transform_segments=segments, fit_segments=self._fit_segments)

        result_df = df
        for transform in self._differencing_transforms:
            result_df = transform._transform(result_df)
        return result_df

    def _inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply inverse transformation to DataFrame.

        Parameters
        ----------
        df:
            DataFrame to apply inverse transform.

        Returns
        -------
        result:
            transformed DataFrame.

        Raises
        ------
        ValueError:
            if transform isn't fitted
        NotImplementedError:
            if there are segments that weren't present during training
        ValueError:
            if inverse transform is applied not to full train nor to test that goes after train
        ValueError:
            if inverse transform is applied to test that goes after train with gap
        """
        self._check_is_fitted()
        if not self.inplace:
            return df

        segments = df.columns.get_level_values("segment").unique().tolist()
        check_new_segments(transform_segments=segments, fit_segments=self._fit_segments)

        result_df = df
        for transform in self._differencing_transforms[::-1]:
            result_df = transform._inverse_transform(result_df)
        return result_df

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes ``order`` parameter. Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "order": IntDistribution(low=1, high=2),
        }
