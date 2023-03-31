from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from etna.datasets import TSDataset


def get_residuals(forecast_df: pd.DataFrame, ts: "TSDataset") -> "TSDataset":
    """Get residuals for further analysis.

    Parameters
    ----------
    forecast_df:
        forecasted dataframe with timeseries data
    ts:
        dataset of timeseries that has answers to forecast

    Returns
    -------
    new_ts: TSDataset
        TSDataset with residuals in forecasts

    Raises
    ------
    KeyError:
        if segments of ``forecast_df`` and ``ts`` aren't the same

    Notes
    -----
    Transforms are taken as is from ``ts``.
    """
    from etna.datasets import TSDataset

    # find the residuals
    true_df = ts[forecast_df.index, :, :]
    if set(ts.segments) != set(forecast_df.columns.get_level_values("segment").unique()):
        raise KeyError("Segments of `ts` and `forecast_df` should be the same")
    true_df.loc[:, pd.IndexSlice[ts.segments, "target"]] -= forecast_df.loc[:, pd.IndexSlice[ts.segments, "target"]]

    # make TSDataset
    new_ts = TSDataset(df=true_df, freq=ts.freq)
    new_ts.known_future = ts.known_future
    new_ts._regressors = ts.regressors
    new_ts.df_exog = ts.df_exog
    return new_ts
