import math
import warnings
from typing import TYPE_CHECKING
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from etna.loggers import tslogger

if TYPE_CHECKING:
    from etna.transforms.base import Transform

TTimestamp = Union[str, pd.Timestamp]


class TSDataset:
    """TSDataset is the main class to handle your time series data.
    It prepares the series for exploration analyzing, implements feature generation with Transforms
    and generation of future points.

    Notes
    -----
    TSDataset supports custom indexing and slicing method.
    It maybe done through these interface: TSDataset[timestamp, segment, column]
    If at the start of the period dataset contains NaN those timestamps will be removed.

    Examples
    --------
    >>> from etna.datasets import generate_const_df
    >>> df = generate_const_df(periods=30, start_time="2021-06-01", n_segments=2, scale=1)
    >>> df_ts_format = TSDataset.to_dataset(df)
    >>> ts = TSDataset(df_ts_format, "D")
    >>> ts["2021-06-01":"2021-06-07", "segment_0", "target"]
    timestamp
    2021-06-01    1.0
    2021-06-02    1.0
    2021-06-03    1.0
    2021-06-04    1.0
    2021-06-05    1.0
    2021-06-06    1.0
    2021-06-07    1.0
    Freq: D, Name: (segment_0, target), dtype: float64

    >>> from etna.datasets import generate_ar_df
    >>> pd.options.display.float_format = '{:,.2f}'.format
    >>> df_to_forecast = generate_ar_df(100, start_time="2021-01-01", n_segments=1)
    >>> df_regressors = generate_ar_df(120, start_time="2021-01-01", n_segments=5)
    >>> df_regressors = df_regressors.pivot(index="timestamp", columns="segment").reset_index()
    >>> df_regressors.columns = ["timestamp"] + [f"regressor_{i}" for i in range(5)]
    >>> df_regressors["segment"] = "segment_0"
    >>> df_to_forecast = TSDataset.to_dataset(df_to_forecast)
    >>> df_regressors = TSDataset.to_dataset(df_regressors)
    >>> tsdataset = TSDataset(df=df_to_forecast, freq="D", df_exog=df_regressors)
    >>> tsdataset.df.head(5)
    segment      segment_0
    feature    regressor_0 regressor_1 regressor_2 regressor_3 regressor_4 target
    timestamp
    2021-01-01        1.62       -0.02       -0.50       -0.56        0.52   1.62
    2021-01-02        1.01       -0.80       -0.81        0.38       -0.60   1.01
    2021-01-03        0.48        0.47       -0.81       -1.56       -1.37   0.48
    2021-01-04       -0.59        2.44       -2.21       -1.21       -0.69  -0.59
    2021-01-05        0.28        0.58       -3.07       -1.45        0.77   0.28
    """

    idx = pd.IndexSlice

    def __init__(self, df: pd.DataFrame, freq: str, df_exog: Optional[pd.DataFrame] = None):
        """Init TSDataset.

        Parameters
        ----------
        df:
            dataframe with timeseries
        freq:
            frequency of timestamp in df
        df_exog:
            dataframe with exogenous data;
            if the series is known in the future features' names should start with prefix 'regressor_`.
        """
        self.raw_df = df.copy(deep=True)
        self.raw_df.index = pd.to_datetime(self.raw_df.index)
        self.freq = freq
        self.df_exog = None

        self.raw_df.index = pd.to_datetime(self.raw_df.index)

        try:
            inferred_freq = pd.infer_freq(self.raw_df.index)
        except ValueError:
            warnings.warn("TSDataset freq can't be inferred")
            inferred_freq = None

        if inferred_freq != self.freq:
            warnings.warn(
                f"You probably set wrong freq. Discovered freq in you data is {inferred_freq}, you set {self.freq}"
            )

        self.raw_df = self.raw_df.asfreq(self.freq)

        self.df = self.raw_df.copy(deep=True)

        if df_exog is not None:
            self.df_exog = df_exog.copy(deep=True)
            self.df_exog.index = pd.to_datetime(self.df_exog.index)
            self.df = self._merge_exog(self.df)

        self.transforms: Optional[Sequence["Transform"]] = None
        self._update_regressors()

    def transform(self, transforms: Sequence["Transform"]):
        """Apply given transform to the data."""
        self._check_endings()
        self.transforms = transforms
        for transform in self.transforms:
            tslogger.log(f"Transform {transform.__class__.__name__} is applied to dataset")
            self.df = transform.transform(self.df)
        self._update_regressors()

    def fit_transform(self, transforms: Sequence["Transform"]):
        """Fit and apply given transforms to the data."""
        self._check_endings()
        self.transforms = transforms
        for transform in self.transforms:
            tslogger.log(f"Transform {transform.__class__.__name__} is applied to dataset")
            self.df = transform.fit_transform(self.df)
        self._update_regressors()

    def __repr__(self):
        return self.df.__repr__()

    def _repr_html_(self):
        return self.df._repr_html_()

    def __getitem__(self, item):
        if isinstance(item, slice) or isinstance(item, str):
            df = self.df.loc[self.idx[item]]
        elif len(item) == 2 and item[0] is Ellipsis:
            df = self.df.loc[self.idx[:], self.idx[:, item[1]]]
        elif len(item) == 2 and item[1] is Ellipsis:
            df = self.df.loc[self.idx[item[0]]]
        else:
            df = self.df.loc[self.idx[item[0]], self.idx[item[1], item[2]]]
        first_valid_idx = df.first_valid_index()
        df = df.loc[first_valid_idx:]
        return df

    def make_future(self, future_steps: int) -> "TSDataset":
        """Return new TSDataset with future steps.

        Parameters
        ----------
        future_steps:
            number of timestamp in the future to build features for.

        Returns
        -------
        :
            dataset with features in the future.

        Examples
        --------
        >>> from etna.datasets import generate_const_df
        >>> df = generate_const_df(
        ...    periods=30, start_time="2021-06-01",
        ...    n_segments=2, scale=1
        ... )
        >>> df_regressors = pd.DataFrame({
        ...     "timestamp": list(pd.date_range("2021-06-01", periods=40))*2,
        ...     "regressor_1": np.arange(80), "regressor_2": np.arange(80) + 5,
        ...     "segment": ["segment_0"]*40 + ["segment_1"]*40
        ... })
        >>> df_ts_format = TSDataset.to_dataset(df)
        >>> df_regressors_ts_format = TSDataset.to_dataset(df_regressors)
        >>> ts = TSDataset(df_ts_format, "D", df_exog=df_regressors_ts_format)
        >>> ts.make_future(4)
        segment      segment_0                      segment_1
        feature    regressor_1 regressor_2 target regressor_1 regressor_2 target
        timestamp
        2021-07-01          30          35    nan          70          75    nan
        2021-07-02          31          36    nan          71          76    nan
        2021-07-03          32          37    nan          72          77    nan
        2021-07-04          33          38    nan          73          78    nan
        """
        max_date_in_dataset = self.df.index.max()
        future_dates = pd.date_range(
            start=max_date_in_dataset, periods=future_steps + 1, freq=self.freq, closed="right"
        )

        new_index = self.raw_df.index.append(future_dates)
        df = self.raw_df.reindex(new_index)
        df.index.name = "timestamp"

        if self.df_exog is not None:
            df = self._merge_exog(df)

            # check if we have enough values in regressors
            if self.regressors:
                for segment in self.segments:
                    regressors_index = self.df_exog.loc[:, pd.IndexSlice[segment, self.regressors]].index
                    if not np.all(future_dates.isin(regressors_index)):
                        warnings.warn(
                            f"Some regressors don't have enough values in segment {segment}, "
                            f"NaN-s will be used for missing values"
                        )

        if self.transforms is not None:
            for transform in self.transforms:
                df = transform.transform(df)

        future_dataset = df.tail(future_steps).copy(deep=True)
        future_dataset = future_dataset.sort_index(axis=1, level=(0, 1))
        future_ts = TSDataset(future_dataset, freq=self.freq)
        future_ts.transforms = self.transforms
        future_ts.df_exog = self.df_exog
        return future_ts

    @staticmethod
    def _check_regressors(df: pd.DataFrame, df_exog: pd.DataFrame):
        """Check that regressors in df_exog begin not later than in df and end later than in df."""
        df_segments = df.columns.get_level_values("segment")
        for segment in df_segments:
            target_min = df[segment]["target"].first_valid_index()
            target_min = pd.NaT if target_min is None else target_min
            target_max = df[segment]["target"].last_valid_index()
            target_max = pd.NaT if target_max is None else target_max
            exog_regressor_columns = [x for x in set(df_exog[segment].columns) if x.startswith("regressor")]
            if len(exog_regressor_columns) == 0:
                continue
            exog_series_min = df_exog[segment][exog_regressor_columns].first_valid_index()
            exog_series_min = pd.NaT if exog_series_min is None else exog_series_min
            exog_series_max = df_exog[segment][exog_regressor_columns].last_valid_index()
            exog_series_max = pd.NaT if exog_series_max is None else exog_series_max
            if target_min < exog_series_min:
                raise ValueError(
                    f"All the regressor series should start not later than corresponding 'target'."
                    f"Series of segment {segment} have not enough history: "
                    f"{target_min} < {exog_series_min}."
                )
            if target_max >= exog_series_max:
                raise ValueError(
                    f"All the regressor series should finish later than corresponding 'target'."
                    f"Series of segment {segment} have not enough history: "
                    f"{target_max} >= {exog_series_max}."
                )

    def _merge_exog(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_regressors(df=df, df_exog=self.df_exog)
        df = pd.concat((df, self.df_exog), axis=1).loc[df.index].sort_index(axis=1, level=(0, 1))
        return df

    def _check_endings(self):
        """Check that all targets ends at the same timestamp."""
        max_index = self.df.index.max()
        if np.any(pd.isna(self.df.loc[max_index, pd.IndexSlice[:, "target"]])):
            raise ValueError(f"All segments should end at the same timestamp")

    def inverse_transform(self):
        """Apply inverse transform method of transforms to the data.
        Applied in reversed order.
        """
        if self.transforms is not None:
            for transform in reversed(self.transforms):
                self.df = transform.inverse_transform(self.df)

    @property
    def segments(self) -> List[str]:
        """Get list of all segments in dataset.

        Examples
        --------
        >>> from etna.datasets import generate_const_df
        >>> df = generate_const_df(
        ...    periods=30, start_time="2021-06-01",
        ...    n_segments=2, scale=1
        ... )
        >>> df_ts_format = TSDataset.to_dataset(df)
        >>> ts = TSDataset(df_ts_format, "D")
        >>> ts.segments
        ['segment_0', 'segment_1']
        """
        return self.df.columns.get_level_values("segment").unique().tolist()

    def _update_regressors(self):
        result = set()
        for column in self.columns.get_level_values("feature"):
            if column.startswith("regressor"):
                result.add(column)
        self._regressors = list(result)

    @property
    def regressors(self) -> List[str]:
        """Get list of all regressors across all segments in dataset.

        Examples
        --------
        >>> from etna.datasets import generate_const_df
        >>> df = generate_const_df(
        ...    periods=30, start_time="2021-06-01",
        ...    n_segments=2, scale=1
        ... )
        >>> df_ts_format = TSDataset.to_dataset(df)
        >>> regressors_timestamp = pd.date_range(start="2021-06-01", periods=50)
        >>> df_regressors_1 = pd.DataFrame(
        ...     {"timestamp": regressors_timestamp, "regressor_1": 1, "segment": "segment_0"}
        ... )
        >>> df_regressors_2 = pd.DataFrame(
        ...     {"timestamp": regressors_timestamp, "regressor_1": 2, "segment": "segment_1"}
        ... )
        >>> df_exog = pd.concat([df_regressors_1, df_regressors_2], ignore_index=True)
        >>> df_exog_ts_format = TSDataset.to_dataset(df_exog)
        >>> ts = TSDataset(df_ts_format, df_exog=df_exog_ts_format, freq="D")
        >>> ts.regressors
        ['regressor_1']
        """
        return self._regressors

    def plot(
        self, n_segments: int = 10, column: str = "target", segments: Optional[Sequence[str]] = None, seed: int = 1
    ):
        """Plot of random or chosen segments.

        Parameters
        ----------
        n_segments:
            number of random segments to plot
        column:
            feature to plot
        segments:
            segments to plot
        seed:
            seed for local random state
        """
        if not segments:
            segments = self.segments
        k = min(n_segments, len(segments))
        columns_num = min(2, k)
        rows_num = math.ceil(k / columns_num)
        _, ax = plt.subplots(rows_num, columns_num, figsize=(20, 5 * rows_num), squeeze=False)
        ax = ax.ravel()
        rnd_state = np.random.RandomState(seed)
        for i, segment in enumerate(sorted(rnd_state.choice(segments, size=k, replace=False))):
            df_slice = self[:, segment, column]
            ax[i].plot(df_slice.index, df_slice.values)
            ax[i].set_title(segment)

        plt.show()

    @staticmethod
    def to_flatten(df: pd.DataFrame) -> pd.DataFrame:
        """Return pandas DataFrame with flatten index.

        Parameters
        ----------
        df:
            DataFrame in ETNA format.

        Returns
        -------
        pd.DataFrame
            with TSDataset data

        Examples
        --------
        >>> from etna.datasets import generate_const_df
        >>> df = generate_const_df(
        ...    periods=30, start_time="2021-06-01",
        ...    n_segments=2, scale=1
        ... )
        >>> df.head(5)
            timestamp    segment  target
        0  2021-06-01  segment_0    1.00
        1  2021-06-02  segment_0    1.00
        2  2021-06-03  segment_0    1.00
        3  2021-06-04  segment_0    1.00
        4  2021-06-05  segment_0    1.00
        >>> df_ts_format = TSDataset.to_dataset(df)
        >>> TSDataset.to_flatten(df_ts_format).head(5)
           timestamp  target    segment
        0 2021-06-01     1.0  segment_0
        1 2021-06-02     1.0  segment_0
        2 2021-06-03     1.0  segment_0
        3 2021-06-04     1.0  segment_0
        4 2021-06-05     1.0  segment_0
        """
        aggregator_list = []
        category = []
        segments = df.columns.get_level_values("segment").unique().tolist()
        for segment in segments:
            if df[segment].select_dtypes(include=["category"]).columns.to_list():
                category.extend(df[segment].select_dtypes(include=["category"]).columns.to_list())
            aggregator_list.append(df[segment].copy())
            aggregator_list[-1]["segment"] = segment
        df = pd.concat(aggregator_list)
        df = df.reset_index()
        category = list(set(category))
        df[category] = df[category].astype("category")
        df.columns.name = None
        return df

    def to_pandas(self, flatten: bool = False) -> pd.DataFrame:
        """Return pandas DataFrame.

        Parameters
        ----------
        flatten:
            If False return pd.DataFrame with multiindex
            if True with flatten index

        Returns
        -------
        pd.DataFrame
            with TSDataset data

        Examples
        --------
        >>> from etna.datasets import generate_const_df
        >>> df = generate_const_df(
        ...    periods=30, start_time="2021-06-01",
        ...    n_segments=2, scale=1
        ... )
        >>> df.head(5)
            timestamp    segment  target
        0  2021-06-01  segment_0    1.00
        1  2021-06-02  segment_0    1.00
        2  2021-06-03  segment_0    1.00
        3  2021-06-04  segment_0    1.00
        4  2021-06-05  segment_0    1.00
        >>> df_ts_format = TSDataset.to_dataset(df)
        >>> ts = TSDataset(df_ts_format, "D")
        >>> ts.to_pandas(True).head(5)
           timestamp  target    segment
        0 2021-06-01     1.0  segment_0
        1 2021-06-02     1.0  segment_0
        2 2021-06-03     1.0  segment_0
        3 2021-06-04     1.0  segment_0
        4 2021-06-05     1.0  segment_0
        >>> ts.to_pandas(False).head(5)
        segment    segment_0 segment_1
        feature       target    target
        timestamp
        2021-06-01      1.00      1.00
        2021-06-02      1.00      1.00
        2021-06-03      1.00      1.00
        2021-06-04      1.00      1.00
        2021-06-05      1.00      1.00
        """
        if not flatten:
            return self.df.copy()
        return self.to_flatten(self.df)

    @staticmethod
    def to_dataset(df: pd.DataFrame) -> pd.DataFrame:
        """Convert pandas dataframe to ETNA Dataset format.

        Parameters
        ----------
        df:
            DataFrame with columns ["timestamp", "segment"]. Other columns considered features.

        Examples
        --------
        >>> from etna.datasets import generate_const_df
        >>> df = generate_const_df(
        ...    periods=30, start_time="2021-06-01",
        ...    n_segments=2, scale=1
        ... )
        >>> df.head(5)
           timestamp    segment  target
        0 2021-06-01  segment_0    1.00
        1 2021-06-02  segment_0    1.00
        2 2021-06-03  segment_0    1.00
        3 2021-06-04  segment_0    1.00
        4 2021-06-05  segment_0    1.00
        >>> df_ts_format = TSDataset.to_dataset(df)
        >>> df_ts_format.head(5)
        segment    segment_0 segment_1
        feature       target    target
        timestamp
        2021-06-01      1.00      1.00
        2021-06-02      1.00      1.00
        2021-06-03      1.00      1.00
        2021-06-04      1.00      1.00
        2021-06-05      1.00      1.00

        >>> df_regressors = pd.DataFrame({
        ...     "timestamp": pd.date_range("2021-01-01", periods=10),
        ...     "regressor_1": np.arange(10), "regressor_2": np.arange(10) + 5,
        ...     "segment": ["segment_0"]*10
        ... })
        >>> TSDataset.to_dataset(df_regressors).head(5)
        segment      segment_0
        feature    regressor_1 regressor_2
        timestamp
        2021-01-01           0           5
        2021-01-02           1           6
        2021-01-03           2           7
        2021-01-04           3           8
        2021-01-05           4           9
        """
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        feature_columns = df.columns.tolist()
        feature_columns.remove("timestamp")
        feature_columns.remove("segment")
        df = df.pivot(index="timestamp", columns="segment")
        df = df.reorder_levels([1, 0], axis=1)
        df.columns.names = ["segment", "feature"]
        df = df.sort_index(axis=1, level=(0, 1))
        return df

    def _find_all_borders(
        self,
        train_start: Optional[TTimestamp],
        train_end: Optional[TTimestamp],
        test_start: Optional[TTimestamp],
        test_end: Optional[TTimestamp],
        test_size: Optional[int],
    ) -> Tuple[TTimestamp, TTimestamp, TTimestamp, TTimestamp]:
        """Find borders for train_test_split if some values wasn't specified."""
        if test_end is not None and test_start is not None and test_size is not None:
            warnings.warn(
                "test_size, test_start and test_end cannot be applied at the same time. test_size will be ignored"
            )

        if test_end is None:
            if test_start is not None and test_size is not None:
                test_start_idx = self.df.index.get_loc(test_start)
                if test_start_idx + test_size > len(self.df.index):
                    raise ValueError(
                        f"test_size is {test_size}, but only {len(self.df.index) - test_start_idx} available with your test_start"
                    )
                test_end_defined = self.df.index[test_start_idx + test_size]
            elif test_size is not None and train_end is not None:
                test_start_idx = self.df.index.get_loc(train_end)
                test_start = self.df.index[test_start_idx + 1]
                test_end_defined = self.df.index[test_start_idx + test_size]
            else:
                test_end_defined = self.df.index.max()
        else:
            test_end_defined = test_end

        if train_start is None:
            train_start_defined = self.df.index.min()
        else:
            train_start_defined = train_start

        if train_end is None and test_start is None and test_size is None:
            raise ValueError("At least one of train_end, test_start or test_size should be defined")

        if test_size is None:
            if train_end is None:
                test_start_idx = self.df.index.get_loc(test_start)
                train_end_defined = self.df.index[test_start_idx - 1]
            else:
                train_end_defined = train_end

            if test_start is None:
                train_end_idx = self.df.index.get_loc(train_end)
                test_start_defined = self.df.index[train_end_idx + 1]
            else:
                test_start_defined = test_start
        else:
            if test_start is None:
                test_start_idx = self.df.index.get_loc(test_end_defined)
                test_start_defined = self.df.index[test_start_idx - test_size + 1]
            else:
                test_start_defined = test_start

            if train_end is None:
                test_start_idx = self.df.index.get_loc(test_start_defined)
                train_end_defined = self.df.index[test_start_idx - 1]
            else:
                train_end_defined = train_end

        if np.datetime64(test_start_defined) < np.datetime64(train_end_defined):
            raise ValueError("The beginning of the test goes before the end of the train")

        return train_start_defined, train_end_defined, test_start_defined, test_end_defined

    def train_test_split(
        self,
        train_start: Optional[TTimestamp] = None,
        train_end: Optional[TTimestamp] = None,
        test_start: Optional[TTimestamp] = None,
        test_end: Optional[TTimestamp] = None,
        test_size: Optional[int] = None,
    ) -> Tuple["TSDataset", "TSDataset"]:
        """Split given df with train-test timestamp indices or size of test set.
        In case of inconsistencies between test_size and (test_start, test_end), test_size is ignored

        Parameters
        ----------
        train_start:
            start timestamp of new train dataset, if None first timestamp is used
        train_end:
            end timestamp of new train dataset, if None previous to test_start timestamp is used
        test_start:
            start timestamp of new test dataset, if None next to train_end timestamp is used
        test_end:
            end timestamp of new test dataset, if None last timestamp is used
        test_size:
            number of timestamps to use in test set

        Returns
        -------
        train, test:
            generated datasets

        Examples
        --------
        >>> from etna.datasets import generate_ar_df
        >>> pd.options.display.float_format = '{:,.2f}'.format
        >>> df = generate_ar_df(100, start_time="2021-01-01", n_segments=3)
        >>> df = TSDataset.to_dataset(df)
        >>> ts = TSDataset(df, "D")
        >>> train_ts, test_ts = ts.train_test_split(
        ...     train_start="2021-01-01", train_end="2021-02-01",
        ...     test_start="2021-02-02", test_end="2021-02-07"
        ... )
        >>> train_ts.df.tail(5)
        segment    segment_0 segment_1 segment_2
        feature       target    target    target
        timestamp
        2021-01-28     -2.06      2.03      1.51
        2021-01-29     -2.33      0.83      0.81
        2021-01-30     -1.80      1.69      0.61
        2021-01-31     -2.49      1.51      0.85
        2021-02-01     -2.89      0.91      1.06
        >>> test_ts.df.head(5)
        segment    segment_0 segment_1 segment_2
        feature       target    target    target
        timestamp
        2021-02-02     -3.57     -0.32      1.72
        2021-02-03     -4.42      0.23      3.51
        2021-02-04     -5.09      1.02      3.39
        2021-02-05     -5.10      0.40      2.15
        2021-02-06     -6.22      0.92      0.97
        """
        train_start_defined, train_end_defined, test_start_defined, test_end_defined = self._find_all_borders(
            train_start, train_end, test_start, test_end, test_size
        )

        if pd.Timestamp(test_end_defined) > self.df.index.max():
            warnings.warn(f"Max timestamp in df is {self.df.index.max()}.")
        if pd.Timestamp(train_start_defined) < self.df.index.min():
            warnings.warn(f"Min timestamp in df is {self.df.index.min()}.")

        train_df = self.df[train_start_defined:train_end_defined][self.raw_df.columns]  # type: ignore
        train_raw_df = self.raw_df[train_start_defined:train_end_defined]  # type: ignore
        train = TSDataset(df=train_df, df_exog=self.df_exog, freq=self.freq)
        train.raw_df = train_raw_df

        test_df = self.df[test_start_defined:test_end_defined][self.raw_df.columns]  # type: ignore
        test_raw_df = self.raw_df[train_start_defined:test_end_defined]  # type: ignore
        test = TSDataset(df=test_df, df_exog=self.df_exog, freq=self.freq)
        test.raw_df = test_raw_df

        return train, test

    @property
    def index(self) -> pd.core.indexes.datetimes.DatetimeIndex:
        """Return TSDataset timestamp index.

        Returns
        -------
        pd.core.indexes.datetimes.DatetimeIndex
            timestamp index of TSDataset
        """
        return self.df.index

    @property
    def columns(self) -> pd.core.indexes.multi.MultiIndex:
        """Return columns of self.df.

        Returns
        -------
        pd.core.indexes.multi.MultiIndex
            multiindex of dataframe with target and features.
        """
        return self.df.columns

    @property
    def loc(self) -> pd.core.indexing._LocIndexer:
        """Return self.df.loc method.

        Returns
        -------
        pd.core.indexing._LocIndexer
            dataframe with self.df.loc[...]
        """
        return self.df.loc

    def isnull(self) -> pd.DataFrame:
        """Return dataframe with flag that means if the correspondent object in self.df is null.

        Returns
        -------
        pd.Dataframe
            is_null dataframe
        """
        return self.df.isnull()

    def head(self, n_rows: int = 5) -> pd.DataFrame:
        """Return the first `n` rows.

        Mimics pandas method.

        This function returns the first `n` rows for the object based
        on position. It is useful for quickly testing if your object
        has the right type of data in it.
        For negative values of `n`, this function returns all rows except
        the last `n` rows, equivalent to ``df[:-n]``.

        Parameters
        ----------
        n_rows:
            number of rows to select.

        Returns
        -------
        pd.DataFrame
            the first `n` rows or 5 by default.
        """
        return self.df.head(n_rows)

    def tail(self, n_rows: int = 5) -> pd.DataFrame:
        """Return the last `n` rows.

        Mimics pandas method.

        This function returns last `n` rows from the object based on
        position. It is useful for quickly verifying data, for example,
        after sorting or appending rows.
        For negative values of `n`, this function returns all rows except
        the first `n` rows, equivalent to ``df[n:]``.

        Parameters
        ----------
        n_rows:
            number of rows to select.

        Returns
        -------
        pd.DataFrame
            the last `n` rows or 5 by default.

        """
        return self.df.tail(n_rows)

    def describe(
        self,
        percentiles: Optional[List[float]] = None,
        include: Optional[Union[str, List[np.dtype]]] = None,
        exclude: Optional[Union[str, List[np.dtype]]] = None,
        datetime_is_numeric: bool = False,
    ) -> pd.DataFrame:
        """Generate descriptive statistics.

        Mimics pandas method.

        Descriptive statistics include those that summarize the central
        tendency, dispersion and shape of a
        dataset's distribution, excluding ``NaN`` values.

        Parameters
        ----------
        percentiles : list-like of numbers, optional
            The percentiles to include in the output. All should
            fall between 0 and 1. The default is
            ``[.25, .5, .75]``, which returns the 25th, 50th, and
            75th percentiles.
        include : 'all', list-like of dtypes or None (default), optional
            A white list of data types to include in the result. Ignored
            for ``Series``. Here are the options:
            - 'all' : All columns of the input will be included in the output.
            - A list-like of dtypes : Limits the results to the
              provided data types.
              To limit the result to numeric types submit
              ``numpy.number``. To limit it instead to object columns submit
              the ``numpy.object`` data type. Strings
              can also be used in the style of
              ``select_dtypes`` (e.g. ``df.describe(include=['O'])``). To
              select pandas categorical columns, use ``'category'``
            - None (default) : The result will include all numeric columns.
        exclude : list-like of dtypes or None (default), optional,
            A black list of data types to omit from the result. Ignored
            for ``Series``. Here are the options:
            - A list-like of dtypes : Excludes the provided data types
              from the result. To exclude numeric types submit
              ``numpy.number``. To exclude object columns submit the data
              type ``numpy.object``. Strings can also be used in the style of
              ``select_dtypes`` (e.g. ``df.describe(include=['O'])``). To
              exclude pandas categorical columns, use ``'category'``
            - None (default) : The result will exclude nothing.
        datetime_is_numeric : bool, default False
            Whether to treat datetime dtypes as numeric. This affects statistics
            calculated for the column. For DataFrame input, this also
            controls whether datetime columns are included by default.

        Returns
        -------
        pd.DataFrame
            Summary statistics of the TSDataset provided.

        """
        return self.df.describe(percentiles, include, exclude, datetime_is_numeric)
