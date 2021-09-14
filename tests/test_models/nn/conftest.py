import pandas as pd
import pytest


@pytest.fixture
def weekly_period_df(n_repeats=15):
    segment_1 = [7.0, 7.0, 7.0, 7.0, 7.0, 3.0, 1.0]
    segment_2 = [7.0, 7.0, 7.0, 4.0, 1.0, 7.0, 7.0]
    ts_range = list(pd.date_range("2020-01-03", freq="1D", periods=n_repeats * len(segment_1)))
    df = pd.DataFrame(
        {
            "timestamp": ts_range * 2,
            "target": segment_1 * n_repeats + segment_2 * n_repeats,
            "segment": ["segment_1"] * n_repeats * len(segment_1) + ["segment_2"] * n_repeats * len(segment_2),
        }
    )
    return df
