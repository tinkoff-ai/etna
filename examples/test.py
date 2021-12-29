# %%

import warnings

warnings.filterwarnings("ignore")

# %%

import pandas as pd
from etna.datasets import TSDataset

# %%

original_df = pd.read_csv("data/monthly-australian-wine-sales.csv")
original_df["timestamp"] = pd.to_datetime(original_df["month"])
original_df["target"] = original_df["sales"]
original_df.drop(columns=["month", "sales"], inplace=True)
original_df["segment"] = "main"
original_df.head()
df = TSDataset.to_dataset(original_df)
ts = TSDataset(df=df, freq="MS")
ts.plot()

# %%

from etna.pipeline import Pipeline
from etna.models import NaiveModel, SeasonalMovingAverageModel, CatBoostModelMultiSegment
from etna.transforms import LagTransform
from etna.metrics import MAE, MSE, SMAPE, MAPE

HORIZON = 3
N_FOLDS = 5

# %%

naive_pipeline = Pipeline(model=NaiveModel(lag=12), transforms=[], horizon=HORIZON)
seasonalma_pipeline = Pipeline(
    model=SeasonalMovingAverageModel(window=5, seasonality=12), transforms=[], horizon=HORIZON
)
catboost_pipeline = Pipeline(
    model=CatBoostModelMultiSegment(),
    transforms=[LagTransform(lags=[6, 7, 8, 9, 10, 11, 12], in_column="target")],
    horizon=HORIZON,
)
pipeline_names = ["naive", "moving average", "catboost"]
pipelines = [naive_pipeline, seasonalma_pipeline, catboost_pipeline]

from etna.ensembles import VotingEnsemble

voting_ensemble = VotingEnsemble(pipelines=pipelines, weights=[1, 9, 4], n_jobs=4)

voting_ensamble_result = voting_ensemble.backtest(
    ts=ts, metrics=[MAE(), MSE(), SMAPE(), MAPE()], n_folds=N_FOLDS, aggregate_metrics=True, n_jobs=2
)
