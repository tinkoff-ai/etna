import pathlib
import tempfile
from copy import deepcopy
from typing import Tuple

import pandas as pd

from etna.datasets import TSDataset
from etna.transforms import Transform


def get_loaded_transform(transform: Transform) -> Transform:
    with tempfile.TemporaryDirectory() as dir_path_str:
        dir_path = pathlib.Path(dir_path_str)
        path = dir_path.joinpath("dummy.zip")
        transform.save(path)
        loaded_transform = deepcopy(transform).load(path)
    return loaded_transform


def assert_transformation_equals_loaded_original(transform: Transform, ts: TSDataset) -> Tuple[Transform, Transform]:
    transform.fit(ts)
    loaded_transform = get_loaded_transform(transform)
    ts_1 = deepcopy(ts)
    ts_2 = deepcopy(ts)

    ts_1.transform([transform])
    ts_2.transform([loaded_transform])

    pd.testing.assert_frame_equal(ts_1.to_pandas(), ts_2.to_pandas())

    return transform, loaded_transform
