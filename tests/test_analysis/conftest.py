import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from etna.datasets import TSDataset
from etna.datasets import duplicate_data


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close()


@pytest.fixture
def exog_and_target_dfs():
    seg = ["a"] * 30 + ["b"] * 30
    time = list(pd.date_range("2020-01-01", "2021-01-01")[:30])
    timestamps = time * 2
    target = np.arange(60)
    df = pd.DataFrame({"segment": seg, "timestamp": timestamps, "target": target})
    ts = TSDataset.to_dataset(df)

    cast = ["1.1"] * 10 + ["2"] * 9 + [None] + ["56.1"] * 10
    no_cast = ["1.1"] * 10 + ["two"] * 10 + ["56.1"] * 10
    none = [1] * 10 + [2] * 10 + [56.1] * 10
    none[10] = None
    df = pd.DataFrame(
        {
            "timestamp": time,
            "exog1": np.arange(100, 70, -1),
            "exog2": np.sin(np.arange(30) / 10),
            "exog3": np.exp(np.arange(30)),
            "cast": cast,
            "no_cast": no_cast,
            "none": none,
        }
    )
    df["cast"] = df["cast"].astype("category")
    df["no_cast"] = df["no_cast"].astype("category")
    df_exog = duplicate_data(df, segments=["a", "b"])
    return ts, df_exog
