import pandas as pd

from etna.datasets.tsdataset import TSDataset
from etna.models import ProphetModel


def test_run(new_format_df):
    df = new_format_df

    ts = TSDataset(df, "1d")

    model = ProphetModel()
    model.fit(ts)
    future_ts = ts.make_future(3)
    model.forecast(future_ts)
    if not future_ts.isnull().values.any():
        assert True
    else:
        assert False


def test_run_with_reg(new_format_df, new_format_exog):
    df = new_format_df
    exog = new_format_exog
    exog.columns = pd.MultiIndex.from_arrays(
        [exog.columns.get_level_values("segment").unique().tolist(), ["regressor_exog", "regressor_exog"]]
    )

    ts = TSDataset(df, "1d", df_exog=exog)

    model = ProphetModel()
    model.fit(ts)
    future_ts = ts.make_future(3)
    model.forecast(future_ts)
    if not future_ts.isnull().values.any():
        assert True
    else:
        assert False
