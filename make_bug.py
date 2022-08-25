import numpy as np
import pandas as pd

from etna.datasets import generate_ar_df, TSDataset


def test_loc_setitem_indexer_differently_ordered():
    mi = pd.MultiIndex.from_product([["a", "b"], [0, 1]])
    df = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], index=mi)

    indexer = ("a", [1, 0])

    df.loc[indexer, :] = np.array([[9, 10], [11, 12]])
    expected = pd.DataFrame([[11, 12], [9, 10], [5, 6], [7, 8]], index=mi)
    pd.testing.assert_frame_equal(df, expected)


def test_loc_getitem_index_differently_ordered_slice_none():
    df = pd.DataFrame(
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        index=[["a", "a", "b", "b"], [1, 2, 1, 2]],
        columns=["a", "b"],
    )
    result = df.loc[(slice(None), [2, 1]), :]
    expected = pd.DataFrame(
        [[3, 4], [7, 8], [1, 2], [5, 6]],
        index=[["a", "b", "a", "b"], [2, 2, 1, 1]],
        columns=["a", "b"],
    )
    pd.testing.assert_frame_equal(result, expected)


def test_on_wide_format():
    df = generate_ar_df(periods=100, start_time="2020-01-01", n_segments=3)
    df_wide = TSDataset.to_dataset(df)

    df_exog = df.copy()
    df_exog = df_exog.rename(columns={"target": "exog_1"})
    df_exog["exog_1"] = df_exog["exog_1"] + 1
    df_exog["exog_2"] = df_exog["exog_1"] + 1
    df_exog["exog_3"] = df_exog["exog_2"] + 1
    df_exog_wide = TSDataset.to_dataset(df_exog)

    ts = TSDataset(df=df_wide, df_exog=df_exog_wide, freq="D")
    df = ts.df

    values = df.loc[:, pd.IndexSlice[:, :]]
    x = 10


def main():
    test_loc_setitem_indexer_differently_ordered()
    test_loc_getitem_index_differently_ordered_slice_none()


if __name__ == "__main__":
    main()
