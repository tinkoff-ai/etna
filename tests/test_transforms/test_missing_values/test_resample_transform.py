import pytest

from etna.transforms.missing_values import ResampleWithDistributionTransform
from tests.test_transforms.utils import assert_transformation_equals_loaded_original


def test_fail_on_incompatible_freq(incompatible_freq_ts):
    resampler = ResampleWithDistributionTransform(
        in_column="exog", inplace=True, distribution_column="target", out_column=None
    )
    with pytest.raises(ValueError, match="Can not infer in_column frequency!"):
        _ = resampler.fit(incompatible_freq_ts)


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
    resampler.fit(ts)
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
    resampled_df = resampler.fit_transform(daily_exog_ts).to_pandas()
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
    future = daily_exog_ts.make_future(3, transforms=[resampler])
    expected_future = expected_resampled_ts.make_future(3)
    assert future.df.equals(expected_future.df)


def test_fit_transform_with_nans(daily_exog_ts_diff_endings):
    resampler = ResampleWithDistributionTransform(
        in_column="regressor_exog", inplace=True, distribution_column="target"
    )
    _ = resampler.fit_transform(daily_exog_ts_diff_endings)


@pytest.mark.filterwarnings("ignore: Regressors info might be incorrect.")
@pytest.mark.parametrize(
    "inplace, in_column_regressor, out_column, expected_regressors",
    [
        (True, False, None, []),
        (False, False, "output_regressor", []),
        (False, True, "output_regressor", ["output_regressor"]),
    ],
)
def test_get_regressors_info(
    daily_exog_ts, inplace, in_column_regressor, out_column, expected_regressors, in_column="regressor_exog"
):
    daily_exog_ts = daily_exog_ts["ts"]
    if in_column_regressor:
        daily_exog_ts._regressors.append(in_column)
    else:
        daily_exog_ts._regressors.remove(in_column)
    resampler = ResampleWithDistributionTransform(
        in_column=in_column, inplace=inplace, distribution_column="target", out_column=out_column
    )
    resampler.fit(daily_exog_ts)
    regressors_info = resampler.get_regressors_info()
    assert sorted(regressors_info) == sorted(expected_regressors)


@pytest.mark.parametrize(
    "inplace,out_column",
    (
        [
            (True, None),
            (False, "resampled_exog"),
        ]
    ),
)
def test_save_load(inplace, out_column, daily_exog_ts):
    daily_exog_ts = daily_exog_ts["ts"]
    transform = ResampleWithDistributionTransform(
        in_column="regressor_exog", inplace=inplace, distribution_column="target", out_column=out_column
    )
    assert_transformation_equals_loaded_original(transform=transform, ts=daily_exog_ts)


def test_get_regressors_info_not_fitted():
    transform = ResampleWithDistributionTransform(in_column="regressor_exog", distribution_column="target")
    with pytest.raises(ValueError, match="Fit the transform to get the correct regressors info!"):
        _ = transform.get_regressors_info()


def test_params_to_tune():
    transform = ResampleWithDistributionTransform(in_column="regressor_exog", distribution_column="target")
    assert len(transform.params_to_tune()) == 0
