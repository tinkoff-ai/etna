import numpy as np
import pytest
from statsmodels.tsa.holtwinters.results import HoltWintersResultsWrapper

from etna.datasets import TSDataset
from etna.datasets import generate_const_df
from etna.metrics import MAE
from etna.models import HoltModel
from etna.models import HoltWintersModel
from etna.models import SimpleExpSmoothingModel
from etna.pipeline import Pipeline


@pytest.fixture
def const_ts():
    """Create a constant dataset with little noise."""
    rng = np.random.default_rng(42)
    df = generate_const_df(start_time="2020-01-01", periods=100, freq="D", n_segments=3, scale=5)
    df["target"] += rng.normal(loc=0, scale=0.05, size=df.shape[0])
    return TSDataset(df=TSDataset.to_dataset(df), freq="D")


@pytest.mark.parametrize(
    "model",
    [
        HoltWintersModel(),
        HoltModel(),
        SimpleExpSmoothingModel(),
    ],
)
def test_holt_winters_simple(model, example_tsds):
    """Test that Holt-Winters' models make predictions in simple case."""
    horizon = 7
    model.fit(example_tsds)
    future_ts = example_tsds.make_future(future_steps=horizon)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)

    assert not res.isnull().values.any()
    assert len(res) == 14


@pytest.mark.parametrize(
    "model",
    [
        HoltWintersModel(),
        HoltModel(),
        SimpleExpSmoothingModel(),
    ],
)
def test_holt_winters_with_exog_warning(model, example_reg_tsds):
    """Test that Holt-Winters' models make predictions with exog with warning."""
    horizon = 7
    model.fit(example_reg_tsds)
    future_ts = example_reg_tsds.make_future(future_steps=horizon)
    with pytest.warns(UserWarning, match="This model does not work with exogenous features and regressors"):
        res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)

    assert not res.isnull().values.any()
    assert len(res) == 14


@pytest.mark.parametrize(
    "model",
    [
        HoltWintersModel(),
        HoltModel(),
        SimpleExpSmoothingModel(),
    ],
)
def test_sanity_const_df(model, const_ts):
    """Test that Holt-Winters' models works good with almost constant dataset."""
    horizon = 7
    train_ts, test_ts = const_ts.train_test_split(test_size=horizon)
    pipeline = Pipeline(model=model, horizon=horizon)
    pipeline.fit(train_ts)
    future_ts = pipeline.forecast()

    mae = MAE(mode="macro")
    mae_value = mae(y_true=test_ts, y_pred=future_ts)
    assert mae_value < 0.05


@pytest.mark.parametrize(
    "etna_model_class",
    (
        HoltModel,
        HoltWintersModel,
        SimpleExpSmoothingModel,
    ),
)
def test_get_model_before_training(etna_model_class):
    """Check that get_model method throws an error if per-segment model is not fitted yet."""
    etna_model = etna_model_class()
    with pytest.raises(ValueError, match="Can not get the dict with base models, the model is not fitted!"):
        _ = etna_model.get_model()


@pytest.mark.parametrize(
    "etna_model_class,expected_class",
    (
        (HoltModel, HoltWintersResultsWrapper),
        (HoltWintersModel, HoltWintersResultsWrapper),
        (SimpleExpSmoothingModel, HoltWintersResultsWrapper),
    ),
)
def test_get_model_after_training(example_tsds, etna_model_class, expected_class):
    """Check that get_model method returns dict of objects of SARIMAX class."""
    pipeline = Pipeline(model=etna_model_class())
    pipeline.fit(ts=example_tsds)
    models_dict = pipeline.model.get_model()
    assert isinstance(models_dict, dict)
    for segment in example_tsds.segments:
        assert isinstance(models_dict[segment], expected_class)
