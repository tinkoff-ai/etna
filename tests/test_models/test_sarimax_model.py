import numpy as np

from etna.datasets import TSDataset
from etna.metrics import MAE
from etna.models import SARIMAXModel
from etna.transforms import TheilSenTrendTransform


def test_sarimax_forecaster_run(example_tsds):
    """
    Given: I have dataframe with 2 segments
    When:
    Then: I get 7 periods per dataset as a forecast
    """

    horizon = 7
    model = SARIMAXModel()
    model.fit(example_tsds)
    future_ts = example_tsds.make_future(future_steps=horizon)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)

    assert not res.isnull().values.any()
    assert len(res) == 14


def test_sarimax_forecaster_run_with_reg(example_reg_tsds):
    """
    Given: I have dataframe with 2 segments
    When:
    Then: I get 7 periods per dataset as a forecast
    """
    horizon = 7
    model = SARIMAXModel()
    model.fit(example_reg_tsds)
    future_ts = example_reg_tsds.make_future(future_steps=horizon)
    res = model.forecast(future_ts)
    res = res.to_pandas(flatten=True)

    assert not res.isnull().values.any()
    assert len(res) == 14


def test_compare_sarimax_vanilla_reg(example_reg_tsds):
    horizon = 24
    example_tsds = TSDataset(example_reg_tsds[:, :, "target"], freq="D")
    train, test = example_tsds.train_test_split(
        train_start=None, train_end="2020-01-31", test_start="2020-02-01", test_end="2020-02-24"
    )
    model = SARIMAXModel()
    model.fit(train)
    future_ts = train.make_future(future_steps=horizon)
    vanilla_result = model.forecast(future_ts)

    train, test = example_reg_tsds.train_test_split(
        train_start=None, train_end="2020-01-31", test_start="2020-02-01", test_end="2020-02-24"
    )
    prep = TheilSenTrendTransform(in_column="target")
    train.fit_transform([prep])
    model = SARIMAXModel()
    model.fit(train)
    future_ts = train.make_future(future_steps=horizon)
    reg_result = model.forecast(future_ts)

    van_acc = np.array(list(MAE()(test, vanilla_result).values()))
    reg_acc = np.array(list(MAE()(test, reg_result).values()))

    assert np.all(van_acc < reg_acc)
