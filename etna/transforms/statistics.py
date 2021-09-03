from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional

import pandas as pd

from etna.transforms.base import Transform


class WindowStatisticsTransform(Transform, ABC):
    """WindowStatisticsTransform handles computation of statistical features on windows."""

    default_out_postfix = "statistics"

    def __init__(
        self,
        window: int,
        in_column: str,
        seasonality: int = 1,
        min_periods: int = 1,
        offset: int = 0,
        out_postfix: Optional[str] = None,
        fillna: float = 0,
        **kwargs,
    ):
        """Init WindowStatisticsTransform.

        Parameters
        ----------
        window: int
            size of window to aggregate
        seasonality: int
            seasonality of lags to compute window's aggregation with
        min_periods: int
            min number of targets in window to compute aggregation; if there is less than min_periods number of targets
            return None
        offset: int
            value of offset to start feature computation
        out_postfix: str, optional
            postfix to add to result column name; if not given, uses default_out_postfix
        fillna: float
            value to fill results NaNs with
        """
        self.window = window
        self.seasonality = seasonality
        self.min_periods = min_periods
        self.offset = offset
        self.out_postfix = out_postfix or self.default_out_postfix
        self.fillna = fillna
        self.kwargs = kwargs
        self.min_required_len = max(self.min_periods - 1, 0) * self.seasonality + 1
        self.history = self.window * self.seasonality
        self.in_column = in_column

    def fit(self, *args) -> "WindowStatisticsTransform":
        """Fits transform."""
        return self

    def _get_required_lags(self, series: pd.Series) -> pd.Series:
        """Get lags according to given seasonality."""
        return pd.Series(series.values[::-1][:: self.seasonality])

    @abstractmethod
    def _aggregate_window(self, series: pd.Series) -> float:
        """Aggregate targets from given series."""
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute feature's value.

        Parameters
        ----------
        df: pd.DataFrame
            dataframe to generate features for

        Returns
        -------
        result: pd.DataFrame
            dataframe with results
        """
        features = (
            df.xs(self.in_column, level=1, axis=1)
            .shift(1 + self.offset)
            .rolling(
                window=self.seasonality * self.window if self.window != -1 else len(df) - 1,
                min_periods=self.min_required_len,
            )
            .aggregate(self._aggregate_window)
        )
        features.fillna(value=self.fillna, inplace=True)

        dataframes = []
        for seg in df.columns.get_level_values(0).unique():
            feature = features[seg].rename(f"{self.in_column}_{self.out_postfix}")
            tmp = df[seg].join(feature)
            _idx = tmp.columns.to_frame()
            _idx.insert(0, "segment", seg)
            tmp.columns = pd.MultiIndex.from_frame(_idx)
            dataframes.append(tmp)

        result = pd.concat(dataframes, axis=1).sort_index(axis=1)
        result.columns.names = ["segment", "feature"]
        return result


class MeanTransform(WindowStatisticsTransform):
    """MeanTransform computes average value for given window.

    .. math::
       MeanTransform(x_t) = \\sum_{i=1}^{window}{x_{t - i}\\cdot\\alpha^{i - 1}}
    """

    default_out_postfix = "mean"

    def __init__(
        self,
        window: int,
        in_column: str,
        seasonality: int = 1,
        alpha: float = 1,
        min_periods: int = 1,
        offset: int = 0,
        out_postfix: Optional[str] = None,
        fillna: float = 0,
    ):
        """Init MeanTransform.

        Parameters
        ----------
        window: int
            size of window to aggregate
        seasonality: int
            seasonality of lags to compute window's aggregation with
        alpha: float
            autoregressive coefficient
        min_periods: int
            min number of targets in window to compute aggregation; if there is less than min_periods number of targets
            return None
        offset: int
            value of offset to start feature computation
        out_postfix: str
            postfix to add to result column name
        fillna: float
            value to fill results NaNs with
        """
        super().__init__(
            window=window,
            in_column=in_column,
            seasonality=seasonality,
            min_periods=min_periods,
            offset=offset,
            out_postfix=out_postfix,
            fillna=fillna,
        )
        self.alpha = alpha
        self._alpha_range: Optional[List[float]] = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute feature's value.

        Parameters
        ----------
        df: pd.DataFrame
            dataframe to generate features for

        Returns
        -------
        result: pd.DataFrame
            dataframe with results
        """
        size = self.window if self.window != -1 else len(df) - 1
        self._alpha_range = [self.alpha ** i for i in range(0, size)]
        return super().transform(df=df)

    def _aggregate_window(self, series: pd.Series) -> Optional[float]:
        """Compute weighted average for window series."""
        tmp_series = self._get_required_lags(series)
        size = len(tmp_series)
        tmp = tmp_series * self._alpha_range[-size:]
        return tmp.mean(**self.kwargs)


class StdTransform(WindowStatisticsTransform):
    """StdTransform computes std value for given window.

    Notes
    -----
    Note that pd.Series([1]).std() is np.nan.
    """

    default_out_postfix = "std"

    def _aggregate_window(self, series: pd.Series) -> Optional[float]:
        """Compute std over the series."""
        tmp_series = self._get_required_lags(series)
        return tmp_series.std(**self.kwargs)


class QuantileTransform(WindowStatisticsTransform):
    """QuantileTransform computes quantile value for given window."""

    default_out_postfix = "quantile"

    def __init__(
        self,
        quantile: float,
        window: int,
        in_column: str,
        seasonality: int = 1,
        min_periods: int = 1,
        offset: int = 0,
        out_postfix: Optional[str] = None,
        fillna: float = 0,
    ):
        """Init QuantileTransform.

        Parameters
        ----------
        quantile: float
            quantile to calculate
        window: int
            size of window to aggregate
        seasonality: int
            seasonality of lags to compute window's aggregation with
        min_periods: int
            min number of targets in window to compute aggregation; if there is less than min_periods number of targets
            return None
        offset: int
            value of offset to start feature computation
        out_postfix: str
            postfix to add to result column name
        fillna: float
            value to fill results NaNs with
        """
        self.quantile = quantile
        super().__init__(
            window=window,
            in_column=in_column,
            seasonality=seasonality,
            min_periods=min_periods,
            offset=offset,
            out_postfix=out_postfix,
            fillna=fillna,
        )
        self.out_postfix = f"{self.out_postfix}_{self.quantile}"

    def _aggregate_window(self, series: pd.Series) -> float:
        """Compute quantile over the series."""
        tmp_series = self._get_required_lags(series)
        return tmp_series.quantile(q=self.quantile, **self.kwargs)


class MinTransform(WindowStatisticsTransform):
    """MinTransform computes min value for given window."""

    default_out_postfix = "min"

    def _aggregate_window(self, series: pd.Series) -> float:
        """Compute min over the series."""
        tmp_series = self._get_required_lags(series)
        return tmp_series.min(**self.kwargs)


class MaxTransform(WindowStatisticsTransform):
    """MaxTransform computes max value for given window."""

    default_out_postfix = "max"

    def _aggregate_window(self, series: pd.Series) -> float:
        """Compute max over the series."""
        tmp_series = self._get_required_lags(series)
        return tmp_series.max(**self.kwargs)


class MedianTransform(WindowStatisticsTransform):
    """MedianTransform computes median value for given window."""

    default_out_postfix = "median"

    def _aggregate_window(self, series: pd.Series) -> float:
        """Compute median over the series."""
        tmp_series = self._get_required_lags(series)
        return tmp_series.median(**self.kwargs)


__all__ = [
    "MedianTransform",
    "MaxTransform",
    "MinTransform",
    "QuantileTransform",
    "StdTransform",
    "MeanTransform",
    "WindowStatisticsTransform",
]
