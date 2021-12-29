import pytest

from etna.transforms.nn import PytorchForecastingTransform


@pytest.mark.parametrize("days_offset", [1, 2, 5, 10])
def test_time_idx(days_offset, example_tsds):
    """Check that PytorchForecastingTransform works with different frequencies correctly."""
    df = example_tsds.to_pandas()
    new_df = df.loc[df.index[::days_offset]]

    transform = PytorchForecastingTransform(
        max_encoder_length=3,
        min_encoder_length=3,
        max_prediction_length=3,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["target"],
        static_categoricals=["segment"],
    )
    transform.fit_transform(new_df)

    time_idx = transform.pf_dataset_train.data["time"].tolist()
    expected_len = new_df.shape[0]
    expected_list = list(range(expected_len)) * len(example_tsds.segments)
    assert time_idx == expected_list
