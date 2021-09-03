import pandas as pd
import pytest

from etna.datasets.tsdataset import TSDataset
from etna.models import CatBoostModelMultiSegment
from etna.models import CatBoostModelPerSegment
from etna.transforms.lags import LagTransform


@pytest.mark.parametrize("catboostmodel", [CatBoostModelMultiSegment, CatBoostModelPerSegment])
def test_run(catboostmodel, new_format_df):
    df = new_format_df
    ts = TSDataset(df, "1d")

    lags = LagTransform(lags=[3, 4, 5], in_column="target")

    ts.fit_transform([lags])

    model = catboostmodel()
    model.fit(ts)
    future_ts = ts.make_future(3)
    model.forecast(future_ts)
    if not future_ts.isnull().values.any():
        assert True
    else:
        assert False


@pytest.mark.parametrize("catboostmodel", [CatBoostModelMultiSegment, CatBoostModelPerSegment])
def test_run_with_reg(catboostmodel, new_format_df, new_format_exog):
    df = new_format_df
    exog = new_format_exog
    exog.columns = pd.MultiIndex.from_arrays(
        [exog.columns.get_level_values("segment").unique().tolist(), ["regressor_exog", "regressor_exog"]]
    )

    ts = TSDataset(df, "1d", df_exog=exog)

    lags = LagTransform(lags=[3, 4, 5], in_column="target")
    lags_exog = LagTransform(lags=[3, 4, 5, 6], in_column="regressor_exog")

    ts.fit_transform([lags, lags_exog])

    model = catboostmodel()
    model.fit(ts)
    future_ts = ts.make_future(3)
    model.forecast(future_ts)
    if not future_ts.isnull().values.any():
        assert True
    else:
        assert False
