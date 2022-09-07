import pytest
from sklearn.linear_model import LinearRegression

from etna.datasets.tsdataset import TSDataset
from etna.models.sklearn import SklearnMultiSegmentModel
from etna.models.sklearn import SklearnPerSegmentModel
from etna.transforms import AddConstTransform
from etna.transforms import LagTransform


@pytest.fixture
def ts_with_regressors(example_df):
    transforms = [
        AddConstTransform(in_column="target", value=10, out_column="add_const_target"),
        LagTransform(in_column="target", lags=[2], out_column="lag"),
    ]
    ts = TSDataset(df=TSDataset.to_dataset(example_df), freq="H", known_future=())
    ts.fit_transform(transforms)
    return ts


@pytest.mark.parametrize("model", ([SklearnPerSegmentModel(regressor=LinearRegression())]))
def test_sklearn_persegment_model_saves_regressors(ts_with_regressors, model):
    """Test that SklearnPerSegmentModel saves the list of regressors from dataset on fit."""
    model.fit(ts_with_regressors)
    for segment_model in model._models.values():
        assert sorted(segment_model.regressor_columns) == sorted(ts_with_regressors.regressors)


@pytest.mark.parametrize("model", ([SklearnPerSegmentModel(regressor=LinearRegression())]))
def test_sklearn_persegment_model_regressors_number(ts_with_regressors, model):
    """Test that the number of features used by SklearnPerSegmentModel is the same as the number of regressors."""
    model.fit(ts_with_regressors)
    for segment_model in model._models.values():
        assert len(segment_model.model.coef_) == len(ts_with_regressors.regressors)


@pytest.mark.parametrize("model", ([SklearnMultiSegmentModel(regressor=LinearRegression())]))
def test_sklearn_multisegment_model_saves_regressors(ts_with_regressors, model):
    """Test that SklearnMultiSegmentModel saves the list of regressors from dataset on fit."""
    model.fit(ts_with_regressors)
    assert sorted(model._base_model.regressor_columns) == sorted(ts_with_regressors.regressors)


@pytest.mark.parametrize("model", ([SklearnMultiSegmentModel(regressor=LinearRegression())]))
def test_sklearn_multisegment_model_regressors_number(ts_with_regressors, model):
    """Test that the number of features used by SklearnMultiSegmentModel is the same as the number of regressors."""
    model.fit(ts_with_regressors)
    assert len(model._base_model.model.coef_) == len(ts_with_regressors.regressors)
