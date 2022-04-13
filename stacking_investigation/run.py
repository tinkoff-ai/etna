import json
import pathlib

import click
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from etna.datasets import TSDataset
from etna.ensembles import StackingEnsemble
from etna.ensembles import VotingEnsemble
from etna.metrics import MAE
from etna.metrics import SMAPE
from etna.models import CatBoostModelMultiSegment
from etna.models import HoltWintersModel
from etna.models import NaiveModel
from etna.models import ProphetModel
from etna.models import SARIMAXModel
from etna.pipeline import Pipeline
from etna.transforms import DateFlagsTransform
from etna.transforms import DensityOutliersTransform
from etna.transforms import FourierTransform
from etna.transforms import LagTransform
from etna.transforms import LinearTrendTransform
from etna.transforms import MeanTransform
from etna.transforms import SegmentEncoderTransform
from etna.transforms import TimeSeriesImputerTransform
from etna.transforms import TrendTransform

HORIZON = 14
METRICS = [MAE(), SMAPE()]


def select_by_indices(array, indices):
    return [array[idx] for idx in indices]


transforms = [
    DensityOutliersTransform(in_column="target", distance_coef=3.0),
    TimeSeriesImputerTransform(in_column="target", strategy="forward_fill"),
    LinearTrendTransform(in_column="target"),
    TrendTransform(in_column="target", out_column="trend"),
    LagTransform(in_column="target", lags=list(range(HORIZON, 122)), out_column="target_lag"),
    DateFlagsTransform(week_number_in_month=True, out_column="date_flag"),
    FourierTransform(period=360.25, order=6, out_column="fourier"),
    SegmentEncoderTransform(),
    MeanTransform(in_column=f"target_lag_{HORIZON}", window=12, seasonality=7),
    MeanTransform(in_column=f"target_lag_{HORIZON}", window=7),
]

pipelines_dict = {
    "sarima": Pipeline(
        model=SARIMAXModel(order=(3, 0, 2), seasonal_order=(3, 0, 2, 7)),
        transforms=select_by_indices(transforms, [0, 1]),
        horizon=HORIZON,
    ),
    "prophet": Pipeline(model=ProphetModel(), transforms=select_by_indices(transforms, [0, 1]), horizon=HORIZON),
    "naive_1": Pipeline(model=NaiveModel(lag=1), transforms=select_by_indices(transforms, [0, 1]), horizon=HORIZON),
    "naive_7": Pipeline(model=NaiveModel(lag=7), transforms=select_by_indices(transforms, [0, 1]), horizon=HORIZON),
    "holt_winters": Pipeline(
        model=HoltWintersModel(seasonal="add", seasonal_periods=7),
        transforms=select_by_indices(transforms, [0, 1]),
        horizon=HORIZON,
    ),
    "catboost": Pipeline(model=CatBoostModelMultiSegment(), transforms=transforms, horizon=HORIZON),
}

all_pipelines = [
    pipelines_dict["sarima"],
    pipelines_dict["prophet"],
    pipelines_dict["naive_1"],
    pipelines_dict["naive_7"],
    pipelines_dict["holt_winters"],
    pipelines_dict["catboost"],
]

final_models = {
    "linear": LinearRegression(),
    "ridge": Ridge(),
    "ridge_positive": Ridge(positive=True),
    "ridge_positive_intercept": Ridge(positive=True, fit_intercept=True),
    "lasso": Lasso(),
    "lasso_positive": Lasso(positive=True),
    "catboost": CatBoostRegressor(),
}


@click.command()
@click.option("--input", type=click.Path(), help="Path to config")
@click.option("--output", type=click.Path(), help="Path to save result")
def main(input: pathlib.Path, output: pathlib.Path):
    # Prepare datasets
    df = pd.read_csv("../examples/data/example_dataset.csv")
    df = TSDataset.to_dataset(df)
    ts = TSDataset(df, freq="D")

    # Get config
    with open(input, "r") as inf:
        config = json.load(inf)

    if config["ensemble"] == "none":
        pipeline = pipelines_dict[config["model"]]
    elif config["ensemble"] == "stacking":
        pipeline = StackingEnsemble(
            pipelines=all_pipelines, n_folds=config["n_folds"], final_model=final_models[config["final_model"]]
        )
    elif config["ensemble"] == "voting":
        pipeline = VotingEnsemble(pipelines=all_pipelines, n_folds=config["n_folds"], weights=config["weights"])

    # Run backtest
    metrics_df, forecast_df, fold_info_df = pipeline.backtest(ts=ts, metrics=METRICS, aggregate_metrics=True)
    result = metrics_df.mean().to_dict()

    with open(output, "w") as ouf:
        json.dump(result, ouf)


if __name__ == "__main__":
    main()
