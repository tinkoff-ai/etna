import pandas as pd
import pytest

from etna.transforms import SegmentEncoderTransform


@pytest.fixture
def dummy_df() -> pd.DataFrame:
    df_1 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="1d")})
    df_2 = pd.DataFrame.from_dict({"timestamp": pd.date_range("2021-06-01", "2021-07-01", freq="1d")})
    df_1["segment"] = "Moscow"
    df_1["target"] = 1
    df_2["segment"] = "Omsk"
    df_2["target"] = 2
    classic_df = pd.concat([df_1, df_2], ignore_index=True)

    df = classic_df.pivot(index="timestamp", columns="segment")
    df = df.reorder_levels([1, 0], axis=1)
    df = df.sort_index(axis=1)
    df.columns.names = ["segment", "feature"]
    return df


def test_dummy(dummy_df):
    transform = SegmentEncoderTransform()
    transformed_df = transform.fit_transform(dummy_df)
    assert (
        len(transformed_df.loc[:, pd.IndexSlice[:, "regressor_segment_code"]].columns) == 2
    ), "Number of columns not the same as segments"
    assert len(dummy_df) == len(transformed_df), "Row missing"
