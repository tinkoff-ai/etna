from copy import deepcopy
from unittest.mock import MagicMock

import pandas as pd
import pytest

from etna.models.base import NonPredictionIntervalContextIgnorantAbstractModel
from etna.models.base import NonPredictionIntervalContextRequiredAbstractModel
from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models.base import PredictionIntervalContextRequiredAbstractModel
from etna.pipeline.mixins import ModelPipelinePredictMixin
from etna.transforms import DateFlagsTransform
from etna.transforms import FilterFeaturesTransform


def make_mixin(ts=None, model=None, transforms=(), mock_recreate_ts=True, mock_determine_prediction_size=True):
    if ts is None:
        ts = MagicMock()
    if model is None:
        model = MagicMock(spec=NonPredictionIntervalContextIgnorantAbstractModel)

    mixin = ModelPipelinePredictMixin()
    mixin.ts = ts
    mixin.transforms = transforms
    mixin.model = model
    if mock_recreate_ts:
        mixin._recreate_ts = MagicMock()
    if mock_determine_prediction_size:
        mixin._determine_prediction_size = MagicMock()
    return mixin


@pytest.mark.parametrize("context_size", [0, 3])
@pytest.mark.parametrize(
    "start_timestamp, end_timestamp",
    [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10")),
        (pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-10")),
    ],
)
@pytest.mark.parametrize(
    "transforms",
    [
        [DateFlagsTransform()],
        [FilterFeaturesTransform(exclude=["regressor_exog_weekend"])],
        [DateFlagsTransform(), FilterFeaturesTransform(exclude=["regressor_exog_weekend"])],
    ],
)
def test_prepare_ts(context_size, start_timestamp, end_timestamp, transforms, example_reg_tsds):
    ts = example_reg_tsds
    initial_ts = deepcopy(ts)
    model = MagicMock()
    model.context_size = context_size
    mixin = make_mixin(ts=ts, transforms=transforms, model=model, mock_recreate_ts=False)

    ts.fit_transform(transforms)
    recreated_ts = mixin._recreate_ts(start_timestamp=start_timestamp, end_timestamp=end_timestamp)

    expected_start_timestamp = max(example_reg_tsds.index[0], start_timestamp - pd.Timedelta(days=model.context_size))
    assert recreated_ts.index[0] == expected_start_timestamp
    assert recreated_ts.index[-1] == end_timestamp
    assert recreated_ts.regressors == initial_ts.regressors
    pd.testing.assert_frame_equal(recreated_ts.df, initial_ts.df[expected_start_timestamp:end_timestamp])


@pytest.mark.parametrize(
    "start_timestamp, end_timestamp, expected_prediction_size",
    [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01"), 1),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), 2),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10"), 10),
        (pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-10"), 6),
    ],
)
def test_determine_prediction_size(start_timestamp, end_timestamp, expected_prediction_size, example_tsds):
    ts = example_tsds
    mixin = make_mixin(ts=ts, mock_determine_prediction_size=False)

    prediction_size = mixin._determine_prediction_size(start_timestamp=start_timestamp, end_timestamp=end_timestamp)

    assert prediction_size == expected_prediction_size


@pytest.mark.parametrize(
    "start_timestamp, end_timestamp",
    [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10")),
        (pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-10")),
    ],
)
def test_predict_recreate_ts_called(start_timestamp, end_timestamp, example_tsds):
    mixin = make_mixin()

    _ = mixin._predict(
        start_timestamp=start_timestamp, end_timestamp=end_timestamp, prediction_interval=False, quantiles=[]
    )

    mixin._recreate_ts.assert_called_once_with(start_timestamp=start_timestamp, end_timestamp=end_timestamp)


@pytest.mark.parametrize(
    "start_timestamp, end_timestamp",
    [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10")),
        (pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-10")),
    ],
)
def test_predict_determine_prediction_size_called(start_timestamp, end_timestamp, example_tsds):
    mixin = make_mixin()

    _ = mixin._predict(
        start_timestamp=start_timestamp, end_timestamp=end_timestamp, prediction_interval=False, quantiles=[]
    )

    mixin._determine_prediction_size.assert_called_once_with(
        start_timestamp=start_timestamp, end_timestamp=end_timestamp
    )


@pytest.mark.parametrize(
    "transforms",
    [
        [DateFlagsTransform()],
        [FilterFeaturesTransform(exclude=["regressor_exog_weekend"])],
    ],
)
def test_predict_apply_transforms(transforms):
    mixin = make_mixin(transforms=transforms)

    _ = mixin._predict(
        start_timestamp=pd.Timestamp("2020-01-01"),
        end_timestamp=pd.Timestamp("2020-01-02"),
        prediction_interval=False,
        quantiles=[],
    )

    expected_ts = mixin._recreate_ts.return_value
    expected_ts.transform.assert_called_once_with(transforms=transforms)


@pytest.mark.parametrize(
    "model_class",
    [NonPredictionIntervalContextIgnorantAbstractModel, NonPredictionIntervalContextRequiredAbstractModel],
)
def test_predict_fail_doesnt_support_prediction_interval(model_class):
    model = MagicMock(spec=model_class)
    mixin = make_mixin(model=model)

    with pytest.raises(
        NotImplementedError, match=f"Model {model.__class__.__name__} doesn't support prediction intervals"
    ):
        _ = mixin._predict(
            start_timestamp=pd.Timestamp("2020-01-01"),
            end_timestamp=pd.Timestamp("2020-01-02"),
            prediction_interval=True,
            quantiles=(0.025, 0.975),
        )


def _check_predict_called(spec, prediction_interval, quantiles, check_keys):
    model = MagicMock(spec=spec)
    mixin = make_mixin(model=model)

    result = mixin._predict(
        start_timestamp=pd.Timestamp("2020-01-01"),
        end_timestamp=pd.Timestamp("2020-01-02"),
        prediction_interval=prediction_interval,
        quantiles=quantiles,
    )

    expected_ts = mixin._recreate_ts.return_value
    expected_prediction_size = mixin._determine_prediction_size.return_value
    called_with_full = dict(
        ts=expected_ts,
        prediction_size=expected_prediction_size,
        prediction_interval=prediction_interval,
        quantiles=quantiles,
    )
    called_with = {key: value for key, value in called_with_full.items() if key in check_keys}
    mixin.model.predict.assert_called_once_with(**called_with)
    assert result == mixin.model.predict.return_value


def test_predict_model_predict_called_non_prediction_interval_context_ignorant():
    _check_predict_called(
        spec=NonPredictionIntervalContextIgnorantAbstractModel,
        prediction_interval=False,
        quantiles=(),
        check_keys=["ts"],
    )


def test_predict_model_predict_called_non_prediction_interval_context_required():
    _check_predict_called(
        spec=NonPredictionIntervalContextRequiredAbstractModel,
        prediction_interval=False,
        quantiles=(),
        check_keys=["ts", "prediction_size"],
    )


@pytest.mark.parametrize("quantiles", [(0.025, 0.975), (0.5,), ()])
@pytest.mark.parametrize("prediction_interval", [False, True])
def test_predict_model_predict_called_prediction_interval_context_ignorant(prediction_interval, quantiles):
    _check_predict_called(
        spec=PredictionIntervalContextIgnorantAbstractModel,
        prediction_interval=False,
        quantiles=(),
        check_keys=["ts", "prediction_interval", "quantiles"],
    )


@pytest.mark.parametrize("quantiles", [(0.025, 0.975), (0.5,), ()])
@pytest.mark.parametrize("prediction_interval", [False, True])
def test_predict_model_predict_called_prediction_interval_context_required(prediction_interval, quantiles):
    _check_predict_called(
        spec=PredictionIntervalContextRequiredAbstractModel,
        prediction_interval=False,
        quantiles=(),
        check_keys=["ts", "prediction_size", "prediction_interval", "quantiles"],
    )
