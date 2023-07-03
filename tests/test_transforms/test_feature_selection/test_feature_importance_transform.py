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

from etna.analysis import ModelRelevanceTable
from etna.analysis import StatisticsRelevanceTable
from etna.datasets import TSDataset
from etna.datasets import generate_ar_df
from etna.models import LinearPerSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import SegmentEncoderTransform
from etna.transforms.feature_selection import TreeFeatureSelectionTransform
from etna.transforms.feature_selection.feature_importance import MRMRFeatureSelectionTransform
from tests.test_transforms.utils import assert_sampling_is_valid
from tests.test_transforms.utils import assert_transformation_equals_loaded_original


@pytest.fixture
def ts_with_regressors():
    num_segments = 3
    df = generate_ar_df(
        start_time="2020-01-01", periods=300, ar_coef=[1], sigma=1, n_segments=num_segments, random_seed=0, freq="D"
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

    # construct TSDataset
    df = df[df["timestamp"] <= timestamp[200]]
    return TSDataset(
        df=TSDataset.to_dataset(df),
        df_exog=TSDataset.to_dataset(df_exog_all_segments),
        freq="D",
        known_future="all",
    )


@pytest.fixture
def ts_with_regressors_and_features(ts_with_regressors):
    le_encoder = SegmentEncoderTransform()
    le_encoder.fit_transform(ts_with_regressors)
    return ts_with_regressors


def test_create_with_unknown_model(ts_with_exog):
    with pytest.raises(ValueError, match="Not a valid option for model: .*"):
        _ = TreeFeatureSelectionTransform(model="unknown", top_k=3, features_to_use="all")


@pytest.mark.parametrize(
    "model",
    [
        "catboost",
        CatBoostRegressor(iterations=10, random_state=42, silent=True),
        CatBoostRegressor(iterations=10, random_state=42, silent=True, cat_features=["segment_code"]),
    ],
)
def test_catboost_with_cat_features(model, ts_with_regressors_and_features):
    """Check that transform with catboost model can work with cat features in a dataset."""
    selector = TreeFeatureSelectionTransform(model=model, top_k=3, features_to_use="all")
    selector.fit_transform(ts_with_regressors_and_features)


@pytest.mark.parametrize(
    "model",
    [
        "random_forest",
        "catboost",
        DecisionTreeRegressor(random_state=42),
        ExtraTreeRegressor(random_state=42),
        RandomForestRegressor(n_estimators=10, random_state=42),
        ExtraTreesRegressor(n_estimators=10, random_state=42),
        GradientBoostingRegressor(n_estimators=10, random_state=42),
        CatBoostRegressor(iterations=10, random_state=42, silent=True),
    ],
)
def test_work_with_non_regressors(ts_with_exog, model):
    selector = TreeFeatureSelectionTransform(model=model, top_k=3, features_to_use="all")
    selector.fit_transform(ts_with_exog)


@pytest.mark.parametrize(
    "model",
    [
        "random_forest",
        "catboost",
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
    ts = ts_with_regressors
    all_regressors = ts_with_regressors.regressors
    selector = TreeFeatureSelectionTransform(model=model, top_k=top_k)
    selector.fit_transform(ts)

    selected_regressors = set(ts.columns.get_level_values("feature")).difference({"target"})
    assert len(selected_regressors) == min(len(all_regressors), top_k)


@pytest.mark.parametrize(
    "model",
    [
        "random_forest",
        "catboost",
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
    """Check that transform doesn't change values of columns."""
    ts = ts_with_regressors
    df_encoded = ts.to_pandas()
    selector = TreeFeatureSelectionTransform(model=model, top_k=top_k)
    df_selected = selector.fit_transform(ts).to_pandas()

    for segment in ts.segments:
        for column in df_selected.columns.get_level_values("feature").unique():
            assert (
                df_selected.loc[:, pd.IndexSlice[segment, column]] == df_encoded.loc[:, pd.IndexSlice[segment, column]]
            ).all()


@pytest.mark.parametrize(
    "model",
    [
        "random_forest",
        "catboost",
        DecisionTreeRegressor(random_state=42),
        ExtraTreeRegressor(random_state=42),
        RandomForestRegressor(n_estimators=10, random_state=42),
        ExtraTreesRegressor(n_estimators=10, random_state=42),
        GradientBoostingRegressor(n_estimators=10, random_state=42),
        CatBoostRegressor(iterations=10, random_state=42, silent=True, cat_features=["segment_code"]),
    ],
)
def test_fails_negative_top_k(model):
    """Check that transform doesn't allow you to set top_k to negative values."""
    with pytest.raises(ValueError, match="positive integer"):
        _ = TreeFeatureSelectionTransform(model=model, top_k=-1)


@pytest.mark.parametrize(
    "model",
    [
        "random_forest",
        "catboost",
        DecisionTreeRegressor(random_state=42),
        ExtraTreeRegressor(random_state=42),
        RandomForestRegressor(n_estimators=10, random_state=42),
        ExtraTreesRegressor(n_estimators=10, random_state=42),
        GradientBoostingRegressor(n_estimators=10, random_state=42),
        CatBoostRegressor(iterations=10, random_state=42, silent=True),
    ],
)
def test_warns_no_regressors(model, example_tsds):
    """Check that transform allows you to fit on dataset with no regressors but warns about it."""
    df = example_tsds.to_pandas()
    selector = TreeFeatureSelectionTransform(model=model, top_k=3)
    with pytest.warns(UserWarning, match="not possible to select features"):
        df_selected = selector.fit_transform(example_tsds).to_pandas()
        assert (df == df_selected).all().all()


@pytest.mark.parametrize(
    "model",
    [
        "random_forest",
        "catboost",
        DecisionTreeRegressor(random_state=42),
        ExtraTreeRegressor(random_state=42),
        RandomForestRegressor(n_estimators=10, random_state=42),
        ExtraTreesRegressor(n_estimators=10, random_state=42),
        GradientBoostingRegressor(n_estimators=10, random_state=42),
        CatBoostRegressor(iterations=700, random_state=42, silent=True),
    ],
)
def test_sanity_selected(model, ts_with_regressors):
    """Check that transform correctly finds meaningful regressors."""
    ts = ts_with_regressors
    selector = TreeFeatureSelectionTransform(model=model, top_k=10)
    df_selected = selector.fit_transform(ts).to_pandas()
    features_columns = df_selected.columns.get_level_values("feature").unique()
    selected_regressors = [column for column in features_columns if column.startswith("regressor_")]
    useful_regressors = [column for column in selected_regressors if "useful" in column]
    assert len(useful_regressors) == 3


@pytest.mark.parametrize(
    "model",
    [
        "random_forest",
        "catboost",
        DecisionTreeRegressor(random_state=42),
        ExtraTreeRegressor(random_state=42),
        RandomForestRegressor(n_estimators=10, random_state=42),
        ExtraTreesRegressor(n_estimators=10, random_state=42),
        GradientBoostingRegressor(n_estimators=10, random_state=42),
        CatBoostRegressor(iterations=500, silent=True, random_state=42, cat_features=["segment_code"]),
    ],
)
def test_sanity_model(model, ts_with_regressors):
    """Check that training with this transform can utilize selected regressors."""
    ts_train, ts_test = ts_with_regressors.train_test_split(test_size=30)
    le_encoder = SegmentEncoderTransform()
    selector = TreeFeatureSelectionTransform(model=model, top_k=8)

    model = LinearPerSegmentModel()
    pipeline = Pipeline(model=model, transforms=[le_encoder, selector], horizon=30)
    pipeline.fit(ts=ts_train)
    ts_forecast = pipeline.forecast()

    for segment in ts_forecast.segments:
        test_target = ts_test[:, segment, "target"]
        forecasted_target = ts_forecast[:, segment, "target"]
        r2 = r2_score(forecasted_target, test_target)
        assert r2 > 0.99


@pytest.mark.parametrize(
    "model",
    [
        "random_forest",
        "catboost",
        DecisionTreeRegressor(random_state=42),
        ExtraTreeRegressor(random_state=42),
        RandomForestRegressor(n_estimators=10, random_state=42),
        ExtraTreesRegressor(n_estimators=10, random_state=42),
        GradientBoostingRegressor(n_estimators=10, random_state=42),
        CatBoostRegressor(iterations=10, random_state=42, silent=True, cat_features=["regressor_exog_weekend"]),
    ],
)
def test_fit_transform_with_nans(model, ts_diff_endings):
    selector = TreeFeatureSelectionTransform(model=model, top_k=10)
    selector.fit_transform(ts_diff_endings)


@pytest.mark.parametrize("fast_redundancy", ([True, False]))
@pytest.mark.parametrize("relevance_table", ([StatisticsRelevanceTable()]))
@pytest.mark.parametrize("top_k", [0, 1, 5, 15, 50])
def test_mrmr_right_len(relevance_table, top_k, ts_with_regressors, fast_redundancy):
    """Check that transform selects exactly top_k regressors."""
    all_regressors = ts_with_regressors.regressors
    ts = ts_with_regressors
    mrmr = MRMRFeatureSelectionTransform(relevance_table=relevance_table, top_k=top_k, fast_redundancy=fast_redundancy)
    df_selected = mrmr.fit_transform(ts).to_pandas()

    selected_regressors = set()
    for column in df_selected.columns.get_level_values("feature"):
        if column.startswith("regressor"):
            selected_regressors.add(column)

    assert len(selected_regressors) == min(len(all_regressors), top_k)


@pytest.mark.parametrize("fast_redundancy", ([True, False]))
@pytest.mark.parametrize("relevance_table", ([ModelRelevanceTable()]))
def test_mrmr_right_regressors(relevance_table, ts_with_regressors, fast_redundancy):
    """Check that transform selects right top_k regressors."""
    ts = ts_with_regressors
    mrmr = MRMRFeatureSelectionTransform(
        relevance_table=relevance_table, top_k=3, model=RandomForestRegressor(), fast_redundancy=fast_redundancy
    )
    df_selected = mrmr.fit_transform(ts).to_pandas()
    selected_regressors = set()
    for column in df_selected.columns.get_level_values("feature"):
        if column.startswith("regressor"):
            selected_regressors.add(column)
    assert set(selected_regressors) == {"regressor_useful_0", "regressor_useful_1", "regressor_useful_2"}


@pytest.mark.parametrize(
    "transform",
    [
        TreeFeatureSelectionTransform(model=DecisionTreeRegressor(random_state=42), top_k=3),
        MRMRFeatureSelectionTransform(
            relevance_table=ModelRelevanceTable(), top_k=3, model=RandomForestRegressor(random_state=42)
        ),
        MRMRFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=3, fast_redundancy=True),
        MRMRFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=3, fast_redundancy=False),
    ],
)
def test_save_load(transform, ts_with_regressors):
    assert_transformation_equals_loaded_original(transform=transform, ts=ts_with_regressors)


@pytest.mark.parametrize(
    "transform",
    [
        TreeFeatureSelectionTransform(model=DecisionTreeRegressor(random_state=42), top_k=3),
        MRMRFeatureSelectionTransform(
            relevance_table=ModelRelevanceTable(), top_k=3, model=RandomForestRegressor(random_state=42)
        ),
        MRMRFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=3, fast_redundancy=True),
        MRMRFeatureSelectionTransform(relevance_table=StatisticsRelevanceTable(), top_k=3, fast_redundancy=False),
    ],
)
def test_params_to_tune(transform, ts_with_regressors):
    ts = ts_with_regressors
    assert len(transform.params_to_tune()) > 0
    assert_sampling_is_valid(transform=transform, ts=ts)
