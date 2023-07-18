import json
import pathlib
import pickle
import zipfile
from copy import deepcopy
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

from etna.models import NaiveModel
from etna.models.base import NonPredictionIntervalContextIgnorantAbstractModel
from etna.models.base import NonPredictionIntervalContextRequiredAbstractModel
from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models.base import PredictionIntervalContextRequiredAbstractModel
from etna.pipeline.mixins import ModelPipelineParamsToTuneMixin
from etna.pipeline.mixins import ModelPipelinePredictMixin
from etna.pipeline.mixins import SaveModelPipelineMixin
from etna.transforms import AddConstTransform
from etna.transforms import DateFlagsTransform
from etna.transforms import FilterFeaturesTransform


def make_mixin(model=None, transforms=(), mock_recreate_ts=True, mock_determine_prediction_size=True):
    if model is None:
        model = MagicMock(spec=NonPredictionIntervalContextIgnorantAbstractModel)

    mixin = ModelPipelinePredictMixin()
    mixin.transforms = transforms
    mixin.model = model
    if mock_recreate_ts:
        mixin._create_ts = MagicMock()
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
def test_predict_mixin_create_ts(context_size, start_timestamp, end_timestamp, transforms, example_reg_tsds):
    ts = example_reg_tsds
    model = MagicMock()
    model.context_size = context_size
    mixin = make_mixin(transforms=transforms, model=model, mock_recreate_ts=False)

    ts.fit_transform(transforms)
    created_ts = mixin._create_ts(ts=ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp)

    expected_start_timestamp = max(example_reg_tsds.index[0], start_timestamp - pd.Timedelta(days=model.context_size))
    assert created_ts.index[0] == expected_start_timestamp
    assert created_ts.index[-1] == end_timestamp
    assert created_ts.regressors == ts.regressors
    expected_df = ts.df[expected_start_timestamp:end_timestamp]
    pd.testing.assert_frame_equal(created_ts.df, expected_df, check_categorical=False)


@pytest.mark.parametrize(
    "start_timestamp, end_timestamp, expected_prediction_size",
    [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01"), 1),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), 2),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10"), 10),
        (pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-10"), 6),
    ],
)
def test_predict_mixin_determine_prediction_size(
    start_timestamp, end_timestamp, expected_prediction_size, example_tsds
):
    ts = example_tsds
    mixin = make_mixin(mock_determine_prediction_size=False)

    prediction_size = mixin._determine_prediction_size(
        ts=ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp
    )

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
def test_predict_mixin_predict_create_ts_called(start_timestamp, end_timestamp, example_tsds):
    ts = MagicMock()
    mixin = make_mixin()

    _ = mixin._predict(
        ts=ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp, prediction_interval=False, quantiles=[]
    )

    mixin._create_ts.assert_called_once_with(ts=ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp)


@pytest.mark.parametrize(
    "start_timestamp, end_timestamp",
    [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10")),
        (pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-10")),
    ],
)
def test_predict_mixin_predict_inverse_transform_called(start_timestamp, end_timestamp, example_tsds):
    ts = MagicMock()
    mixin = make_mixin()

    result = mixin._predict(
        ts=ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp, prediction_interval=False, quantiles=[]
    )

    result.inverse_transform.assert_called_once_with(mixin.transforms)


@pytest.mark.parametrize(
    "start_timestamp, end_timestamp",
    [
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-10")),
        (pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-10")),
    ],
)
def test_predict_mixin_predict_determine_prediction_size_called(start_timestamp, end_timestamp, example_tsds):
    ts = MagicMock()
    mixin = make_mixin()

    _ = mixin._predict(
        ts=ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp, prediction_interval=False, quantiles=[]
    )

    mixin._determine_prediction_size.assert_called_once_with(
        ts=ts, start_timestamp=start_timestamp, end_timestamp=end_timestamp
    )


@pytest.mark.parametrize(
    "model_class",
    [NonPredictionIntervalContextIgnorantAbstractModel, NonPredictionIntervalContextRequiredAbstractModel],
)
def test_predict_mixin_predict_fail_doesnt_support_prediction_interval(model_class):
    ts = MagicMock()
    model = MagicMock(spec=model_class)
    mixin = make_mixin(model=model)

    with pytest.raises(
        NotImplementedError, match=f"Model {model.__class__.__name__} doesn't support prediction intervals"
    ):
        _ = mixin._predict(
            ts=ts,
            start_timestamp=pd.Timestamp("2020-01-01"),
            end_timestamp=pd.Timestamp("2020-01-02"),
            prediction_interval=True,
            quantiles=(0.025, 0.975),
        )


def _check_predict_called(spec, prediction_interval, quantiles, return_components, check_keys):
    ts = MagicMock()
    model = MagicMock(spec=spec)
    mixin = make_mixin(model=model)

    result = mixin._predict(
        ts=ts,
        start_timestamp=pd.Timestamp("2020-01-01"),
        end_timestamp=pd.Timestamp("2020-01-02"),
        prediction_interval=prediction_interval,
        quantiles=quantiles,
        return_components=return_components,
    )

    expected_ts = mixin._create_ts.return_value
    expected_prediction_size = mixin._determine_prediction_size.return_value
    called_with_full = dict(
        ts=expected_ts,
        prediction_size=expected_prediction_size,
        prediction_interval=prediction_interval,
        quantiles=quantiles,
        return_components=return_components,
    )
    called_with = {key: value for key, value in called_with_full.items() if key in check_keys}
    mixin.model.predict.assert_called_once_with(**called_with)
    assert result == mixin.model.predict.return_value


@pytest.mark.parametrize("return_components", [False, True])
def test_predict_mixin_predict_called_non_prediction_interval_context_ignorant(return_components):
    _check_predict_called(
        spec=NonPredictionIntervalContextIgnorantAbstractModel,
        prediction_interval=False,
        quantiles=(),
        return_components=return_components,
        check_keys=["ts", "return_components"],
    )


@pytest.mark.parametrize("return_components", [False, True])
def test_predict_mixin_predict_called_non_prediction_interval_context_required(return_components):
    _check_predict_called(
        spec=NonPredictionIntervalContextRequiredAbstractModel,
        prediction_interval=False,
        quantiles=(),
        return_components=return_components,
        check_keys=["ts", "prediction_size", "return_components"],
    )


@pytest.mark.parametrize("quantiles", [(0.025, 0.975), (0.5,), ()])
@pytest.mark.parametrize("prediction_interval", [False, True])
@pytest.mark.parametrize("return_components", [False, True])
def test_predict_mixin_predict_called_prediction_interval_context_ignorant(
    prediction_interval, quantiles, return_components
):
    _check_predict_called(
        spec=PredictionIntervalContextIgnorantAbstractModel,
        prediction_interval=False,
        quantiles=(),
        return_components=return_components,
        check_keys=["ts", "prediction_interval", "quantiles", "return_components"],
    )


@pytest.mark.parametrize("quantiles", [(0.025, 0.975), (0.5,), ()])
@pytest.mark.parametrize("prediction_interval", [False, True])
@pytest.mark.parametrize("return_components", [False, True])
def test_predict_mixin_predict_called_prediction_interval_context_required(
    prediction_interval, quantiles, return_components
):
    _check_predict_called(
        spec=PredictionIntervalContextRequiredAbstractModel,
        prediction_interval=False,
        quantiles=(),
        return_components=return_components,
        check_keys=["ts", "prediction_size", "prediction_interval", "quantiles", "return_components"],
    )


class Dummy(SaveModelPipelineMixin):
    def __init__(self, a, b, ts, model, transforms):
        self.a = a
        self.b = b
        self.ts = ts
        self.model = model
        self.transforms = transforms


def test_save_mixin_save(example_tsds, tmp_path):
    model = NaiveModel()
    transforms = [AddConstTransform(in_column="target", value=10.0)]
    dummy = Dummy(a=1, b=2, ts=example_tsds, model=model, transforms=transforms)
    dir_path = pathlib.Path(tmp_path)
    path = dir_path / "dummy.zip"

    initial_dummy = deepcopy(dummy)
    initial_transforms = deepcopy(transforms)
    dummy.save(path)

    with zipfile.ZipFile(path, "r") as archive:
        files = archive.namelist()
        assert sorted(files) == sorted(["metadata.json", "object.pkl", "model.zip", "transforms/00000000.zip"])

        with archive.open("metadata.json", "r") as input_file:
            metadata_bytes = input_file.read()
        metadata_str = metadata_bytes.decode("utf-8")
        metadata = json.loads(metadata_str)
        assert sorted(metadata.keys()) == ["class", "etna_version"]
        assert metadata["class"] == "tests.test_pipeline.test_mixins.Dummy"

        with archive.open("object.pkl", "r") as input_file:
            loaded_obj = pickle.load(input_file)
        assert loaded_obj.a == dummy.a
        assert loaded_obj.b == dummy.b

    # check that we didn't break dummy object itself
    assert dummy.a == initial_dummy.a
    assert pickle.dumps(dummy.ts) == pickle.dumps(initial_dummy.ts)
    assert pickle.dumps(dummy.model) == pickle.dumps(initial_dummy.model)
    assert pickle.dumps(dummy.transforms) == pickle.dumps(initial_transforms)


def test_save_mixin_load_fail_file_not_found():
    non_existent_path = pathlib.Path("archive.zip")
    with pytest.raises(FileNotFoundError):
        Dummy.load(non_existent_path)


def test_save_mixin_load_ok_no_ts(example_tsds, recwarn, tmp_path):
    model = NaiveModel()
    transform_values = list(range(1, 11))
    transforms = [AddConstTransform(in_column="target", value=value) for value in transform_values]
    dummy = Dummy(a=1, b=2, ts=example_tsds, model=model, transforms=transforms)
    dir_path = pathlib.Path(tmp_path)
    path = dir_path / "dummy.zip"

    dummy.save(path)
    loaded_dummy = Dummy.load(path)

    assert loaded_dummy.a == dummy.a
    assert loaded_dummy.b == dummy.b
    assert loaded_dummy.ts is None
    assert isinstance(loaded_dummy.model, NaiveModel)
    assert [transform.value for transform in loaded_dummy.transforms] == transform_values
    assert len(recwarn) == 0


def test_save_mixin_load_ok_with_ts(example_tsds, recwarn, tmp_path):
    model = NaiveModel()
    transform_values = list(range(1, 11))
    transforms = [AddConstTransform(in_column="target", value=value) for value in transform_values]
    dummy = Dummy(a=1, b=2, ts=example_tsds, model=model, transforms=transforms)
    dir_path = pathlib.Path(tmp_path)
    path = dir_path / "dummy.zip"

    dummy.save(path)
    loaded_dummy = Dummy.load(path, ts=example_tsds)

    assert loaded_dummy.a == dummy.a
    assert loaded_dummy.b == dummy.b
    assert loaded_dummy.ts is not example_tsds
    pd.testing.assert_frame_equal(loaded_dummy.ts.to_pandas(), dummy.ts.to_pandas())
    assert isinstance(loaded_dummy.model, NaiveModel)
    assert [transform.value for transform in loaded_dummy.transforms] == transform_values
    assert len(recwarn) == 0


@pytest.mark.parametrize(
    "save_version, load_version", [((1, 5, 0), (2, 5, 0)), ((2, 5, 0), (1, 5, 0)), ((1, 5, 0), (1, 3, 0))]
)
@patch("etna.core.mixins.get_etna_version")
def test_save_mixin_load_warning(get_version_mock, save_version, load_version, example_tsds, tmp_path):
    model = NaiveModel()
    transforms = [AddConstTransform(in_column="target", value=10.0)]
    dummy = Dummy(a=1, b=2, ts=example_tsds, model=model, transforms=transforms)
    dir_path = pathlib.Path(tmp_path)
    path = dir_path / "dummy.zip"

    get_version_mock.return_value = save_version
    dummy.save(path)

    save_version_str = ".".join([str(x) for x in save_version])
    load_version_str = ".".join([str(x) for x in load_version])
    with pytest.warns(
        UserWarning,
        match=f"The object was saved under etna version {save_version_str} but running version is {load_version_str}",
    ):
        get_version_mock.return_value = load_version
        _ = Dummy.load(path)


def test_param_to_tune_mixin():
    mixin = ModelPipelineParamsToTuneMixin()
    model = MagicMock()
    model.params_to_tune.return_value = {"alpha": [1, 2, 3], "beta": [4, 5, 6]}
    transform_1 = MagicMock()
    transform_1.params_to_tune.return_value = {"param_1": ["option_1", "option_2"], "param_2": [False, True]}
    transform_2 = MagicMock()
    transform_2.params_to_tune.return_value = {"param_3": [1, 2]}
    mixin.model = model
    mixin.transforms = [transform_1, transform_2]

    obtained_params_to_tune = mixin.params_to_tune()

    expected_params_to_tune = {
        "model.alpha": [1, 2, 3],
        "model.beta": [4, 5, 6],
        "transforms.0.param_1": ["option_1", "option_2"],
        "transforms.0.param_2": [False, True],
        "transforms.1.param_3": [1, 2],
    }
    assert obtained_params_to_tune == expected_params_to_tune
