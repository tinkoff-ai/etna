import pandas as pd
import pytest
from catboost import CatBoostRegressor
from numpy.random import RandomState
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor

from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.models import LinearPerSegmentModel
from etna.pipeline import Pipeline
from etna.transforms.feature_importance import TreeFeatureSelectionTransform


@pytest.fixture
def ts_with_regressors():
    num_segments = 3
    df = generate_ar_df(
        start_time="2020-01-01", periods=300, ar_coef=[1], sigma=1, n_segments=num_segments, random_seed=0
    )

    example_segment = df["segment"].unique()[0]
    timestamp = df[df["segment"] == example_segment]["timestamp"]
    df_exog = pd.DataFrame({"timestamp": timestamp})

    # useless regressors
    num_useless = 12
    df_regressors_useless = generate_ar_df(
        start_time="2020-01-01", periods=300, ar_coef=[1], sigma=1, n_segments=num_useless, random_seed=1, freq="D"
    )
    for i, segment in enumerate(df_regressors_useless["segment"].unique()):
        regressor = df_regressors_useless[df_regressors_useless["segment"] == segment]["target"].values
        df_exog[f"regressor_useless_{i}"] = regressor

    # useful regressors: the same as target but with little noise
    df_regressors_useful = df.copy()
    sampler = RandomState(seed=2).normal
    for i, segment in enumerate(df_regressors_useful["segment"].unique()):
        regressor = df_regressors_useful[df_regressors_useful["segment"] == segment]["target"].values
        noise = sampler(scale=0.05, size=regressor.shape)
        df_exog[f"regressor_useful_{i}"] = regressor + noise

    # construct exog
    classic_exog_list = []
    for segment in df["segment"].unique():
        tmp = df_exog.copy(deep=True)
        tmp["segment"] = segment
        classic_exog_list.append(tmp)

    df_exog_all_segments = pd.concat(classic_exog_list)
    df = df[df["timestamp"] <= timestamp[200]]

    # construct TSDataset
    return TSDataset(df=TSDataset.to_dataset(df), df_exog=TSDataset.to_dataset(df_exog_all_segments), freq="D")


@pytest.mark.parametrize(
    "model",
    [
        DecisionTreeRegressor(random_state=42),
        ExtraTreeRegressor(random_state=42),
        RandomForestRegressor(n_estimators=10, random_state=42),
        ExtraTreesRegressor(n_estimators=10, random_state=42),
        GradientBoostingRegressor(n_estimators=10, random_state=42),
        CatBoostRegressor(iterations=10, random_state=42, silent=True),
    ],
)
@pytest.mark.parametrize("top_k", [0, 1, 5, 15, 50])
def test_selected_top_k_regressors(model, top_k, ts_with_regressors):
    """Check that transform selects exactly top_k regressors if where are this much."""
    df = ts_with_regressors.to_pandas()
    transform = TreeFeatureSelectionTransform(model=model, top_k=top_k)
    all_regressors = ts_with_regressors.regressors
    df_transformed = transform.fit_transform(df)

    selected_regressors = set()
    for column in df_transformed.columns.get_level_values("feature"):
        if column.startswith("regressor"):
            selected_regressors.add(column)

    assert len(selected_regressors) == min(len(all_regressors), top_k)


@pytest.mark.parametrize(
    "model",
    [
        DecisionTreeRegressor(random_state=42),
        ExtraTreeRegressor(random_state=42),
        RandomForestRegressor(n_estimators=10, random_state=42),
        ExtraTreesRegressor(n_estimators=10, random_state=42),
        GradientBoostingRegressor(n_estimators=10, random_state=42),
        CatBoostRegressor(iterations=10, random_state=42, silent=True),
    ],
)
@pytest.mark.parametrize("top_k", [0, 1, 5, 15, 50])
def test_retain_values(model, top_k, ts_with_regressors):
    """Check that transform does not change values of columns."""
    df = ts_with_regressors.to_pandas()
    transform = TreeFeatureSelectionTransform(model=model, top_k=top_k)
    df_transformed = transform.fit_transform(df)

    for segment in ts_with_regressors.segments:
        for column in df_transformed.columns.get_level_values("feature"):
            assert (
                df_transformed.loc[:, pd.IndexSlice[segment, column]] == df.loc[:, pd.IndexSlice[segment, column]]
            ).all()


@pytest.mark.parametrize(
    "model",
    [
        DecisionTreeRegressor(random_state=42),
        ExtraTreeRegressor(random_state=42),
        RandomForestRegressor(n_estimators=10, random_state=42),
        ExtraTreesRegressor(n_estimators=10, random_state=42),
        GradientBoostingRegressor(n_estimators=10, random_state=42),
        CatBoostRegressor(iterations=10, random_state=42, silent=True),
    ],
)
def test_fails_negative_top_k(model, ts_with_regressors):
    """Check that transform don't allow you to set top_k to negative values."""
    with pytest.raises(ValueError, match="positive integer"):
        TreeFeatureSelectionTransform(model=model, top_k=-1)


@pytest.mark.parametrize(
    "model",
    [
        DecisionTreeRegressor(random_state=42),
        ExtraTreeRegressor(random_state=42),
        RandomForestRegressor(n_estimators=10, random_state=42),
        ExtraTreesRegressor(n_estimators=10, random_state=42),
        GradientBoostingRegressor(n_estimators=10, random_state=42),
        CatBoostRegressor(iterations=10, random_state=42, silent=True),
    ],
)
def test_no_regressors(model, example_tsds):
    """Check that transform not fails if there are no regressors.

    In this case train data will be only information about segments.
    """
    df = example_tsds.to_pandas()
    transform = TreeFeatureSelectionTransform(model=model, top_k=3)
    df_transformed = transform.fit_transform(df)

    for segment in example_tsds.segments:
        for column in df_transformed.columns.get_level_values("feature"):
            assert (
                df_transformed.loc[:, pd.IndexSlice[segment, column]] == df.loc[:, pd.IndexSlice[segment, column]]
            ).all()


@pytest.mark.parametrize(
    "model",
    [
        DecisionTreeRegressor(random_state=42),
        ExtraTreeRegressor(random_state=42),
        RandomForestRegressor(n_estimators=10, random_state=42),
        ExtraTreesRegressor(n_estimators=10, random_state=42),
        GradientBoostingRegressor(n_estimators=10, random_state=42),
        CatBoostRegressor(iterations=600, random_state=42, silent=True),
    ],
)
def test_sanity_selected(model, ts_with_regressors):
    """Check that transform correctly finds meaningful regressors."""
    df = ts_with_regressors.to_pandas()
    transform = TreeFeatureSelectionTransform(model=model, top_k=5)
    df_transformed = transform.fit_transform(df)
    features_columns = df_transformed.columns.get_level_values("feature").unique()
    selected_regressors = [column for column in features_columns if column.startswith("regressor_")]
    useful_regressors = [column for column in selected_regressors if "useful" in column]
    assert len(useful_regressors) == 3


@pytest.mark.parametrize(
    "model",
    [
        DecisionTreeRegressor(random_state=42),
        ExtraTreeRegressor(random_state=42),
        RandomForestRegressor(n_estimators=10, random_state=42),
        ExtraTreesRegressor(n_estimators=10, random_state=42),
        GradientBoostingRegressor(n_estimators=10, random_state=42),
        CatBoostRegressor(iterations=500, silent=True, random_state=42),
    ],
)
def test_sanity_model(model, ts_with_regressors):
    """Check that training with this transform cat utilize found regressors."""
    ts_train, ts_test = ts_with_regressors.train_test_split(test_size=30)
    transform = TreeFeatureSelectionTransform(model=model, top_k=6)

    model = LinearPerSegmentModel()
    pipeline = Pipeline(model=model, transforms=[transform], horizon=30)
    pipeline.fit(ts=ts_train)
    ts_forecast = pipeline.forecast()

    for segment in ts_forecast.segments:
        test_target = ts_test[:, segment, "target"]
        forecasted_target = ts_forecast[:, segment, "target"]
        r2 = r2_score(forecasted_target, test_target)
        assert r2 > 0.99
