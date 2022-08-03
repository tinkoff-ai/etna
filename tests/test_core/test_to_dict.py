import pickle
from json import JSONDecoder

import hydra_slayer
import pytest
from ruptures import Binseg
from sklearn.linear_model import LinearRegression

from etna.core import BaseMixin
from etna.ensembles import StackingEnsemble
from etna.ensembles import VotingEnsemble
from etna.metrics import MAE
from etna.models import CatBoostModelPerSegment
from etna.models import LinearPerSegmentModel
from etna.pipeline import Pipeline
from etna.transforms import AddConstTransform
from etna.transforms import ChangePointsTrendTransform
from etna.transforms import LogTransform


@pytest.fixture()
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
    ensemble1 = VotingEnsemble(pipelines=[pipeline1, pipeline2], weights=[0.4, 0.6])
    ensemble2 = StackingEnsemble(pipelines=[pipeline1, pipeline2])
    return ensemble1, ensemble2


class _InvalidParsing(BaseMixin):
    def __init__(self, a: JSONDecoder):
        self.a = a


@pytest.mark.parametrize(
    "target_object",
    [
        AddConstTransform(in_column="target", value=10),
        ChangePointsTrendTransform(
            in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=50
        ),
        CatBoostModelPerSegment(),
        MAE(mode="macro"),
        Pipeline(
            model=CatBoostModelPerSegment(),
            transforms=[
                AddConstTransform(in_column="target", value=10),
                ChangePointsTrendTransform(
                    in_column="target", change_point_model=Binseg(), detrend_model=LinearRegression(), n_bkps=50
                ),
            ],
            horizon=5,
        ),
    ],
)
def test_to_dict_transform(target_object):
    dict_object = target_object.to_dict()
    transformed_object = hydra_slayer.get_from_params(**dict_object)
    assert pickle.dumps(transformed_object) == pickle.dumps(target_object)


def test_ensembles(ensemble_samples):
    ensemble1, ensemble2 = ensemble_samples
    transformed_object_1 = hydra_slayer.get_from_params(**ensemble1.to_dict())
    assert pickle.dumps(transformed_object_1) == pickle.dumps(ensemble1)
    transformed_object_2 = hydra_slayer.get_from_params(**ensemble2.to_dict())
    assert pickle.dumps(transformed_object_2) == pickle.dumps(ensemble2)


def test_warnings():
    with pytest.warns(Warning, match="Some of external objects in input parameters is not instance of BaseEstimator"):
        _ = _InvalidParsing(JSONDecoder()).to_dict()
