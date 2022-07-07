import random

import numpy as np
import pandas as pd
import torch

from etna.analysis import plot_backtest
from etna.datasets import TSDataset
from etna.metrics import MAE, MSE, SMAPE
from etna.models.nn.rnn import RNN
from etna.pipeline import Pipeline
from etna.transforms import LagTransform, StandardScalerTransform

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from etna.loggers import tslogger, WandbLogger


tslogger.add(WandbLogger(project="native_test", tags=["rnn"]))

original_df = pd.read_csv("examples/data/example_dataset.csv")
original_df.head()


df = TSDataset.to_dataset(original_df)
ts = TSDataset(df, freq="D")
ts.head(5)


lags = [10, 11, 12, 13, 14]
pipe = Pipeline(
    model=RNN(
        6,
        trainer_params=dict(max_epochs=5),
        test_batch_size=4,
        encoder_length=14,
        decoder_length=14,
        split_params=dict(train_frac=0.8),
    ),
    transforms=[StandardScalerTransform("target"), LagTransform(in_column="target", lags=lags, out_column="lag")],
    horizon=7,
)


metrics, forecast, fold_info = pipe.backtest(ts, metrics=[SMAPE(), MAE(), MSE()], n_folds=3, n_jobs=1)


plot_backtest(forecast, ts, history_len=20)


score = metrics["SMAPE"].mean()
print(f"Average SMAPE for LSTM: {score:.3f}")
