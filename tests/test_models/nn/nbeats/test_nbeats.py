import pytest

from etna.metrics import MAE
from etna.models.nn import NBeatsGenericModel
from etna.models.nn import NBeatsInterpretableModel
from etna.transforms import StandardScalerTransform
from tests.test_models.utils import assert_model_equals_loaded_original


def run_model_test(model, ts_train, ts_test, horizon):
    model.fit(ts_train)

    future = ts_train.make_future(horizon, tail_steps=model.input_size)
    future = model.forecast(future, prediction_size=horizon)

    mae = MAE("macro")
    return mae(ts_test, future)


@pytest.mark.parametrize(
    "horizon",
    [
        8,
        13,
        15,
    ],
)
def test_interpretable_model_run_weekly_overfit_with_scaler(ts_dataset_weekly_function_with_horizon, horizon):
    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)

    model = NBeatsInterpretableModel(
        input_size=horizon,
        output_size=horizon,
        loss="mae",
        trend_blocks=1,
        trend_layers=1,
        trend_layer_size=64,
        degree_of_polynomial=2,
        seasonality_blocks=1,
        seasonality_layers=1,
        seasonality_layer_size=256,
        lr=0.001,
        num_of_harmonics=1,
    )

    metric = run_model_test(model=model, ts_train=ts_train, ts_test=ts_test, horizon=horizon)
    assert metric < 0.4


@pytest.mark.parametrize(
    "horizon",
    [
        8,
        13,
        15,
    ],
)
def test_generic_model_run_weekly_overfit_with_scaler(ts_dataset_weekly_function_with_horizon, horizon):
    ts_train, ts_test = ts_dataset_weekly_function_with_horizon(horizon)

    model = NBeatsGenericModel(
        input_size=horizon,
        output_size=horizon,
        loss="mae",
        stacks=1,
        layers=4,
        layer_size=128,
        lr=0.001,
    )

    metric = run_model_test(model=model, ts_train=ts_train, ts_test=ts_test, horizon=horizon)
    assert metric < 0.4


@pytest.mark.parametrize(
    "model",
    (
        NBeatsInterpretableModel(
            input_size=6,
            output_size=3,
            loss="mse",
            trend_blocks=1,
            trend_layers=1,
            trend_layer_size=16,
            degree_of_polynomial=2,
            seasonality_blocks=1,
            seasonality_layers=1,
            seasonality_layer_size=16,
            trainer_params=dict(max_epochs=100),
        ),
        NBeatsGenericModel(
            input_size=6,
            output_size=3,
            loss="mse",
            stacks=1,
            layers=2,
            layer_size=16,
            lr=0.001,
            trainer_params=dict(max_epochs=100),
        ),
    ),
)
def test_save_load(example_tsds, model):
    horizon = model.output_size
    std = StandardScalerTransform(in_column="target")
    assert_model_equals_loaded_original(model=model, ts=example_tsds, transforms=[std], horizon=horizon)


@pytest.mark.parametrize(
    "model", (NBeatsInterpretableModel(input_size=6, output_size=3), NBeatsGenericModel(input_size=6, output_size=3))
)
def test_context_size(model, expected=6):
    assert model.context_size == expected
