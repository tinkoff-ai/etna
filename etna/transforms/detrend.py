import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor

from etna.transforms.base import PerSegmentWrapper
from etna.transforms.base import Transform


class LinearTrendBaseTransform(Transform):
    """LinearTrendBaseTransform is a base class that implements trend subtraction and reconstruction feature."""

    is_categorical = False
    is_static = False

    def __init__(self, in_column: str, regressor=None):
        # TODO: add inplace arg
        """
        Create instance of LinearTrendFeature.

        Parameters
        ----------
        regressor: instance of sklearn regressors to predict trend
        """
        self._linear_model = regressor
        self.in_column = in_column

    def fit(self, df: pd.DataFrame):
        """
        Fit regression detrend_model with data from df.

        Parameters
        ----------
        df: pd.Series
            data that regressor should be trained with

        Returns
        -------
            LinearTrendFeature instance with trained regressor
        """
        series_len = len(df)
        x = df.index.to_series()
        if isinstance(type(x.dtype), pd.Timestamp):
            raise ValueError("Your timestamp column has wrong format. Need np.datetime64 or datetime.datetime")
        x = x.apply(lambda ts: ts.timestamp())
        x = x.to_numpy().reshape(series_len, 1)
        y = df[self.in_column].tolist()
        self._linear_model.fit(x, y)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data from df: subtract linear trend found by regressor.

        Parameters
        ----------
        df: pd.Series
            data series to subtract trend from

        Returns
        -------
            pd.Series
                residue after trend subtraction
        """
        result = df.copy()
        series_len = len(df)
        x = pd.to_datetime(df.index.to_series())
        x = x.apply(lambda ts: ts.timestamp())
        x = x.to_numpy().reshape(series_len, 1)
        y = df[self.in_column].values
        trend = self._linear_model.predict(x)
        no_trend_timeseries = y - trend
        result[self.in_column] = no_trend_timeseries
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit regression detrend_model with data from df and subtract the trend from df.

        Parameters
        ----------
        df: pd.Series
            data to train regressor and transform

        Returns
        -------
            residue after trend subtraction
        """
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.Series) -> pd.DataFrame:
        """
        Inverse transformation for trend subtraction: add trend to prediction.

        Parameters
        ----------
        df: pd.Series
            data to transform

        Returns
        -------
            data with reconstructed trend
        """
        result = df.copy()
        series_len = len(df)
        x = pd.to_datetime(df.index.to_series())
        x = x.apply(lambda ts: ts.timestamp())
        x = x.to_numpy().reshape(series_len, 1)
        y = df[self.in_column].values
        trend = self._linear_model.predict(x)
        add_trend_timeseries = y + trend
        result[self.in_column] = add_trend_timeseries
        return result


class _LinearTrendTransform(LinearTrendBaseTransform):
    """LinearTrend use sklearn.linear_model.LinearRegression to find linear trend in data."""

    def __init__(self, in_column: str, *args, **regression_params):
        """Create instance of LinearTrend.

        Parameters
        ----------
        regression_params: Dict[str, Any]
            params that should be used to init LinearRegression
        """
        super().__init__(in_column=in_column, regressor=LinearRegression(**regression_params))


class _TheilSenTrendTransform(LinearTrendBaseTransform):
    """TheilSenTrend use sklearn.linear_model.TheilSenRegressor to find linear trend in data."""

    def __init__(self, in_column, *args, **regression_params):
        """Create instance of TheilSenTrend.

        Parameters
        ----------
        regression_params: Dict[str, Any]
            params that should be used to init TheilSenRegressor
        """
        super().__init__(in_column=in_column, regressor=TheilSenRegressor(**regression_params))


class LinearTrendTransform(PerSegmentWrapper):
    """LinearTrend use sklearn.linear_model.LinearRegression to find linear trend in data."""

    def __init__(self, in_column: str, *args, **regression_params):
        """Create instance of LinearTrend.

        Parameters
        ----------
        regression_params: Dict[str, Any]
            params that should be used to init LinearRegression
        """
        super().__init__(transform=_LinearTrendTransform(in_column, *args, **regression_params))


class TheilSenTrendTransform(PerSegmentWrapper):
    """TheilSenTrend use sklearn.linear_model.TheilSenRegressor to find linear trend in data."""

    def __init__(self, in_column: str, *args, **regression_params):
        """Create instance of TheilSenTrend.

        Parameters
        ----------
        regression_params: Dict[str, Any]
            params that should be used to init TheilSenRegressor
        """
        super().__init__(transform=_TheilSenTrendTransform(in_column, *args, **regression_params))
