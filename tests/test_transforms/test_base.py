import pandas as pd

from etna.transforms import Transform


class DummyTransform(Transform):
    def fit(self, df: pd.DataFrame) -> "DummyTransform":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


def test_default_params_to_tune():
    dummy = DummyTransform()
    assert dummy.params_to_tune() == {}
