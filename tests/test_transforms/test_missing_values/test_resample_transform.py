import pytest

from etna.transforms.missing_values import ResampleWithDistributionTransform


def test_fail_on_incompatible_freq(incompatible_freq_ts):
    resampler = ResampleWithDistributionTransform(
        in_column="exog", inplace=True, distribution_column="target", out_column=None
    )
    with pytest.raises(ValueError, match="Can not infer in_column frequency!"):
        _ = resampler.fit(incompatible_freq_ts.df)


@pytest.mark.parametrize(
    "ts",
    (
        [
            "daily_exog_ts",
            "weekly_exog_same_start_ts",
            "weekly_exog_diff_start_ts",
        ]
    ),
)
def test_fit(ts, request):
    ts = request.getfixturevalue(ts)
    ts, expected_distribution = ts["ts"], ts["distribution"]
    resampler = ResampleWithDistributionTransform(
        in_column="regressor_exog", inplace=True, distribution_column="target", out_column=None
    )
    resampler.fit(ts.df)
    segments = ts.df.columns.get_level_values("segment").unique()
    for segment in segments:
        assert (resampler.segment_transforms[segment].distribution == expected_distribution[segment]).all().all()


@pytest.mark.parametrize(
    "inplace,out_column,expected_resampled_ts",
    (
        [
            (True, None, "inplace_resampled_daily_exog_ts"),
            (False, "resampled_exog", "noninplace_resampled_daily_exog_ts"),
        ]
    ),
)
def test_transform(daily_exog_ts, inplace, out_column, expected_resampled_ts, request):
    daily_exog_ts = daily_exog_ts["ts"]
    expected_resampled_df = request.getfixturevalue(expected_resampled_ts).df
    resampler = ResampleWithDistributionTransform(
        in_column="regressor_exog", inplace=inplace, distribution_column="target", out_column=out_column
    )
    resampled_df = resampler.fit_transform(daily_exog_ts.df)
    assert resampled_df.equals(expected_resampled_df)


@pytest.mark.parametrize(
    "inplace,out_column,expected_resampled_ts",
    (
        [
            (True, None, "inplace_resampled_daily_exog_ts"),
            (False, "resampled_exog", "noninplace_resampled_daily_exog_ts"),
        ]
    ),
)
def test_transform_future(daily_exog_ts, inplace, out_column, expected_resampled_ts, request):
    daily_exog_ts = daily_exog_ts["ts"]
    expected_resampled_ts = request.getfixturevalue(expected_resampled_ts)
    resampler = ResampleWithDistributionTransform(
        in_column="regressor_exog", inplace=inplace, distribution_column="target", out_column=out_column
    )
    daily_exog_ts.fit_transform([resampler])
    future = daily_exog_ts.make_future(3)
    expected_future = expected_resampled_ts.make_future(3)
    assert future.df.equals(expected_future.df)


def test_fit_transform_with_nans(daily_exog_ts_diff_endings):
    resampler = ResampleWithDistributionTransform(
        in_column="regressor_exog", inplace=True, distribution_column="target"
    )
    daily_exog_ts_diff_endings.fit_transform([resampler])
