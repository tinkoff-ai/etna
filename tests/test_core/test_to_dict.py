import json
import pickle

import hydra_slayer
import pytest
from ruptures import Binseg
from sklearn.linear_model import LinearRegression

from etna.core import BaseMixin
from etna.ensembles import StackingEnsemble
from etna.ensembles import VotingEnsemble
from etna.metrics import MAE
from etna.metrics import SMAPE
from etna.models import AutoARIMAModel
from etna.models import CatBoostModelPerSegment
from etna.models import LinearPerSegmentModel
from etna.models.nn import DeepARModel
from etna.pipeline import Pipeline
from etna.transforms import AddConstTransform
from etna.transforms import ChangePointsTrendTransform
from etna.transforms import LogTransform
from etna.transforms import LambdaTransform

def ensemble_samples():
    pipeline1 = Pipeline(
        model=CatBoostModelPerSegment(),
        transforms=[
            AddConstTransform(in_column="target", value=10),
            ChangePointsTrendTransform(
                in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=50
            ),
        ],
        horizon=5,
    )
    pipeline2 = Pipeline(
        model=LinearPerSegmentModel(),
        transforms=[
            ChangePointsTrendTransform(
                in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=50
            ),
            LogTransform(in_column="target"),
        ],
        horizon=5,
    )
    return [pipeline1, pipeline2]


@pytest.mark.parametrize(
    "target_object",
    [
        AddConstTransform(in_column="target", value=10),
        ChangePointsTrendTransform(
            in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=50
        ),
        pytest.param(LambdaTransform(in_column="target", transform_func=lambda x:x-2, inverse_transform_func=lambda x:x+2), marks=pytest.mark.xfail(reason="some bug"))
    ],
)
def test_to_dict_transforms(target_object):
    dict_object = target_object.to_dict()
    transformed_object = hydra_slayer.get_from_params(**dict_object)
    assert json.loads(json.dumps(dict_object)) == dict_object
    assert pickle.dumps(transformed_object) == pickle.dumps(target_object)


@pytest.mark.parametrize(
    "target_model",
    [
        pytest.param(DeepARModel(), marks=pytest.mark.xfail(reason="some bug")),
        LinearPerSegmentModel(),
        CatBoostModelPerSegment(),
        AutoARIMAModel(),
    ],
)
def test_to_dict_models(target_model):
    dict_object = target_model.to_dict()
    transformed_object = hydra_slayer.get_from_params(**dict_object)
    assert json.loads(json.dumps(dict_object)) == dict_object
    assert pickle.dumps(transformed_object) == pickle.dumps(target_model)


@pytest.mark.parametrize(
    "target_object",
    [
        Pipeline(
            model=CatBoostModelPerSegment(),
            transforms=[
                AddConstTransform(in_column="target", value=10),
                ChangePointsTrendTransform(
                    in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=50
                ),
            ],
            horizon=5,
        )
    ],
)
def test_to_dict_pipeline(target_object):
    dict_object = target_object.to_dict()
    transformed_object = hydra_slayer.get_from_params(**dict_object)
    assert json.loads(json.dumps(dict_object)) == dict_object
    assert pickle.dumps(transformed_object) == pickle.dumps(target_object)


@pytest.mark.parametrize("target_object", [MAE(mode="macro"), SMAPE()])
def test_to_dict_metrics(target_object):
    dict_object = target_object.to_dict()
    transformed_object = hydra_slayer.get_from_params(**dict_object)
    assert json.loads(json.dumps(dict_object)) == dict_object
    assert pickle.dumps(transformed_object) == pickle.dumps(target_object)


@pytest.mark.parametrize(
    "target_ensemble",
    [VotingEnsemble(pipelines=ensemble_samples(), weights=[0.4, 0.6]), StackingEnsemble(pipelines=ensemble_samples())],
)
def test_ensembles(target_ensemble):
    dict_object = target_ensemble.to_dict()
    transformed_object = hydra_slayer.get_from_params(**dict_object)
    assert json.loads(json.dumps(dict_object)) == dict_object
    assert pickle.dumps(transformed_object) == pickle.dumps(target_ensemble)


class _Dummy:
    pass


class _InvalidParsing(BaseMixin):
    def __init__(self, a: _Dummy):
        self.a = a


def test_warnings():
    with pytest.warns(Warning, match="Some of external objects in input parameters could be not written in dict"):
        _ = _InvalidParsing(_Dummy()).to_dict()
