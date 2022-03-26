from abc import ABC
from abc import abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from etna.transforms.base import Transform


class WindowStatisticsTransform(Transform, ABC):
    """WindowStatisticsTransform handles computation of statistical features on windows."""

    def __init__(
        self,
        in_column: str,
        out_column: str,
        window: int,
        seasonality: int = 1,
        min_periods: int = 1,
        fillna: float = 0,
        **kwargs,
    ):
        """Init WindowStatisticsTransform.

        Parameters
        ----------
        in_column: str
            name of processed column
        out_column: str
            result column name
        window: int
            size of window to aggregate, if -1 is set all history is used
        seasonality: int
            seasonality of lags to compute window's aggregation with
        min_periods: int
            min number of targets in window to compute aggregation; if there is less than min_periods number of targets
            return None
        fillna: float
            value to fill results NaNs with
        """
        self.in_column = in_column
        self.out_column_name = out_column
        self.window = window
        self.seasonality = seasonality
        self.min_periods = min_periods
        self.fillna = fillna
        self.kwargs = kwargs

    def fit(self, *args) -> "WindowStatisticsTransform":
        """Fits transform."""
        return self

    @abstractmethod
    def _aggregate(self, series: np.ndarray) -> np.ndarray:
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
        history = self.seasonality * self.window if self.window != -1 else len(df)
        segments = sorted(df.columns.get_level_values("segment").unique())

        x = df.loc[pd.IndexSlice[:], pd.IndexSlice[segments, self.in_column]].values[::-1]
        x = np.append(x, np.empty((history - 1, x.shape[1])) * np.nan, axis=0)
        x = np.lib.stride_tricks.sliding_window_view(x, window_shape=(history, 1))[:, :, :: self.seasonality]
        x = np.squeeze(x, axis=-1)

        window = self.window if self.window != -1 else len(df)
        nan_per_window_counts = np.sum(np.isnan(x), axis=2)
        non_nan_per_window_counts = window - nan_per_window_counts

        y = self._aggregate(series=x)
        y[non_nan_per_window_counts < self.min_periods] = np.nan
        y = np.nan_to_num(y, copy=False, nan=self.fillna)[::-1]

        result = df.join(
            pd.DataFrame(y, columns=pd.MultiIndex.from_product([segments, [self.out_column_name]]), index=df.index)
        )
        result = result.sort_index(axis=1)
        return result


class MeanTransform(WindowStatisticsTransform):
    """MeanTransform computes average value for given window.

    .. math::
       MeanTransform(x_t) = \\sum_{i=1}^{window}{x_{t - i}\\cdot\\alpha^{i - 1}}
    """

    def __init__(
        self,
        in_column: str,
        window: int,
        seasonality: int = 1,
        alpha: float = 1,
        min_periods: int = 1,
        fillna: float = 0,
        out_column: Optional[str] = None,
    ):
        """Init MeanTransform.

        Parameters
        ----------
        in_column: str
            name of processed column
        window: int
            size of window to aggregate
        seasonality: int
            seasonality of lags to compute window's aggregation with
        alpha: float
            autoregressive coefficient
        min_periods: int
            min number of targets in window to compute aggregation; if there is less than min_periods number of targets
            return None
        fillna: float
            value to fill results NaNs with
        out_column: str, optional
            result column name. If not given use `self.__repr__()`
        """
        self.window = window
        self.in_column = in_column
        self.seasonality = seasonality
        self.alpha = alpha
        self.min_periods = min_periods
        self._alpha_range: Optional[np.ndarray] = None
        self.fillna = fillna
        self.out_column = out_column
        super().__init__(
            in_column=in_column,
            window=window,
            seasonality=seasonality,
            min_periods=min_periods,
            out_column=self.out_column if self.out_column is not None else self.__repr__(),
            fillna=fillna,
        )

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
        size = self.window if self.window != -1 else len(df)
        self._alpha_range = np.array([self.alpha ** i for i in range(0, size)])
        return super().transform(df=df)

    def _aggregate(self, series: np.ndarray) -> np.ndarray:
        """Compute weighted average for window series."""
        if self._alpha_range is None:
            raise ValueError("Something went wrong generating the alphas!")
        #series = self._alpha_range[np.newaxis, np.newaxis, : series.shape[-1]] * series
        series = np.nanmean(series, axis=2)
        return series


class StdTransform(WindowStatisticsTransform):
    """StdTransform computes std value for given window.

    Notes
    -----
    Note that pd.Series([1]).std() is np.nan.
    """

    def __init__(
        self,
        in_column: str,
        window: int,
        seasonality: int = 1,
        min_periods: int = 1,
        fillna: float = 0,
        out_column: Optional[str] = None,
        ddof: int = 1,
    ):
        """Init StdTransform.

        Parameters
        ----------
        in_column: str
            name of processed column
        window: int
            size of window to aggregate
        seasonality: int
            seasonality of lags to compute window's aggregation with
        min_periods: int
            min number of targets in window to compute aggregation; if there is less than min_periods number of targets
            return None
        fillna: float
            value to fill results NaNs with
        out_column: str, optional
            result column name. If not given use `self.__repr__()`
        ddof:
            delta degrees of freedom; the divisor used in calculations is N - ddof, where N is the number of elements
        """
        self.in_column = in_column
        self.window = window
        self.seasonality = seasonality
        self.min_periods = min_periods
        self.fillna = fillna
        self.out_column = out_column
        self.ddof = ddof
        super().__init__(
            window=window,
            in_column=in_column,
            seasonality=seasonality,
            min_periods=min_periods,
            out_column=self.out_column if self.out_column is not None else self.__repr__(),
            fillna=fillna,
        )

    def _aggregate(self, series: np.ndarray) -> np.ndarray:
        """Compute std over the series."""
        series = np.nanstd(series, axis=2, ddof=self.ddof)
        return series


class QuantileTransform(WindowStatisticsTransform):
    """QuantileTransform computes quantile value for given window."""

    def __init__(
        self,
        in_column: str,
        quantile: float,
        window: int,
        seasonality: int = 1,
        min_periods: int = 1,
        fillna: float = 0,
        out_column: Optional[str] = None,
    ):
        """Init QuantileTransform.

        Parameters
        ----------
        in_column: str
            name of processed column
        quantile: float
            quantile to calculate
        window: int
            size of window to aggregate
        seasonality: int
            seasonality of lags to compute window's aggregation with
        min_periods: int
            min number of targets in window to compute aggregation; if there is less than min_periods number of targets
            return None
        fillna: float
            value to fill results NaNs with
        out_column: str, optional
            result column name. If not given use `self.__repr__()`
        """
        self.in_column = in_column
        self.quantile = quantile
        self.window = window
        self.seasonality = seasonality
        self.min_periods = min_periods
        self.fillna = fillna
        self.out_column = out_column
        super().__init__(
            in_column=in_column,
            window=window,
            seasonality=seasonality,
            min_periods=min_periods,
            out_column=self.out_column if self.out_column is not None else self.__repr__(),
            fillna=fillna,
        )

    def _aggregate(self, series: np.ndarray) -> np.ndarray:
        """Compute quantile over the series."""
        series = np.nanquantile(series, self.quantile, axis=2)
        return series


class MinTransform(WindowStatisticsTransform):
    """MinTransform computes min value for given window."""

    def __init__(
        self,
        in_column: str,
        window: int,
        seasonality: int = 1,
        min_periods: int = 1,
        fillna: float = 0,
        out_column: Optional[str] = None,
    ):
        """Init MinTransform.

        Parameters
        ----------
        in_column: str
            name of processed column
        window: int
            size of window to aggregate
        seasonality: int
            seasonality of lags to compute window's aggregation with
        min_periods: int
            min number of targets in window to compute aggregation; if there is less than min_periods number of targets
            return None
        fillna: float
            value to fill results NaNs with
        out_column: str, optional
            result column name. If not given use `self.__repr__()`
        """
        self.in_column = in_column
        self.window = window
        self.seasonality = seasonality
        self.min_periods = min_periods
        self.fillna = fillna
        self.out_column = out_column
        super().__init__(
            window=window,
            in_column=in_column,
            seasonality=seasonality,
            min_periods=min_periods,
            out_column=self.out_column if self.out_column is not None else self.__repr__(),
            fillna=fillna,
        )

    def _aggregate(self, series: np.ndarray) -> np.ndarray:
        """Compute min over the series."""
        series = np.nanmin(series, axis=2)
        return series


class MaxTransform(WindowStatisticsTransform):
    """MaxTransform computes max value for given window."""

    def __init__(
        self,
        in_column: str,
        window: int,
        seasonality: int = 1,
        min_periods: int = 1,
        fillna: float = 0,
        out_column: Optional[str] = None,
    ):
        """Init MaxTransform.

        Parameters
        ----------
        in_column: str
            name of processed column
        window: int
            size of window to aggregate
        seasonality: int
            seasonality of lags to compute window's aggregation with
        min_periods: int
            min number of targets in window to compute aggregation; if there is less than min_periods number of targets
            return None
        fillna: float
            value to fill results NaNs with
        out_column: str, optional
            result column name. If not given use `self.__repr__()`
        """
        self.in_column = in_column
        self.window = window
        self.seasonality = seasonality
        self.min_periods = min_periods
        self.fillna = fillna
        self.out_column = out_column
        super().__init__(
            window=window,
            in_column=in_column,
            seasonality=seasonality,
            min_periods=min_periods,
            out_column=self.out_column if self.out_column is not None else self.__repr__(),
            fillna=fillna,
        )

    def _aggregate(self, series: np.ndarray) -> np.ndarray:
        """Compute max over the series."""
        series = np.nanmax(series, axis=2)
        return series


class MedianTransform(WindowStatisticsTransform):
    """MedianTransform computes median value for given window."""

    def __init__(
        self,
        in_column: str,
        window: int,
        seasonality: int = 1,
        min_periods: int = 1,
        fillna: float = 0,
        out_column: Optional[str] = None,
    ):
        """Init MedianTransform.

        Parameters
        ----------
        in_column: str
            name of processed column
        window: int
            size of window to aggregate
        seasonality: int
            seasonality of lags to compute window's aggregation with
        min_periods: int
            min number of targets in window to compute aggregation; if there is less than min_periods number of targets
            return None
        fillna: float
            value to fill results NaNs with
        out_column: str, optional
            result column name. If not given use `self.__repr__()`
        """
        self.in_column = in_column
        self.window = window
        self.seasonality = seasonality
        self.min_periods = min_periods
        self.fillna = fillna
        self.out_column = out_column
        super().__init__(
            window=window,
            in_column=in_column,
            seasonality=seasonality,
            min_periods=min_periods,
            out_column=self.out_column if self.out_column is not None else self.__repr__(),
            fillna=fillna,
        )

    def _aggregate(self, series: np.ndarray) -> np.ndarray:
        """Compute median over the series."""
        series = np.nanmedian(series, axis=2)
        return series


class MADTransform(WindowStatisticsTransform):
    """MADTransform computes Mean Absolute Deviation over the window."""

    def __init__(
        self,
        in_column: str,
        window: int,
        seasonality: int = 1,
        min_periods: int = 1,
        fillna: float = 0,
        out_column: Optional[str] = None,
    ):
        """Init MADTransform.

        Parameters
        ----------
        in_column: str
            name of processed column
        window: int
            size of window to aggregate
        seasonality: int
            seasonality of lags to compute window's aggregation with
        min_periods: int
            min number of targets in window to compute aggregation; if there is less than min_periods number of targets
            return None
        fillna: float
            value to fill results NaNs with
        out_column: str, optional
            result column name. If not given use `self.__repr__()`
        """
        self.in_column = in_column
        self.window = window
        self.seasonality = seasonality
        self.min_periods = min_periods
        self.fillna = fillna
        self.out_column = out_column
        super().__init__(
            window=window,
            in_column=in_column,
            seasonality=seasonality,
            min_periods=min_periods,
            out_column=self.out_column if self.out_column is not None else self.__repr__(),
            fillna=fillna,
        )

    def _aggregate(self, series: np.ndarray) -> np.ndarray:
        """Compute median over the series."""
        mean = np.nanmean(series, axis=2)
        ad = np.abs(series - mean[:, :, np.newaxis])
        mad = np.nanmean(ad, axis=2)
        return mad


__all__ = [
    "MedianTransform",
    "MaxTransform",
    "MinTransform",
    "QuantileTransform",
    "StdTransform",
    "MeanTransform",
    "WindowStatisticsTransform",
    "MADTransform",
]
