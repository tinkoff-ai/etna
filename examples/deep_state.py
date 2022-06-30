import random

import numpy as np
import pandas as pd
import torch

from etna.analysis import plot_backtest
from etna.datasets import TSDataset
from etna.metrics import MAE, MSE, SMAPE
from etna.models.nn.deepstate import DeepStateNetwork, SeasonalitySSM
from etna.pipeline import Pipeline
from etna.transforms import SegmentEncoderTransform

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


original_df = pd.read_csv("examples/data/example_dataset.csv")
original_df.head()


df = TSDataset.to_dataset(original_df)
ts = TSDataset(df, freq="D")
ts.head(5)


n_samples = 140
horizon = 7
encoder_length = 28
decoder_length = 14
ssm = SeasonalitySSM(num_seasons=7)


pipe = Pipeline(
    horizon=horizon,
    model=DeepStateNetwork(
        ssm,
        encoder_length=encoder_length,
        decoder_length=decoder_length,
        n_samples=n_samples,
        input_size=1,
        train_batch_size=32,
        test_batch_size=1,
        trainer_kwargs=dict(max_epochs=1),
    ),
    transforms=[SegmentEncoderTransform()],
)


metrics, forecast, fold_info = pipe.backtest(ts, metrics=[SMAPE(), MAE(), MSE()], n_folds=3, n_jobs=1)


plot_backtest(forecast, ts, history_len=20)


score = metrics["SMAPE"].mean()
print(f"Average SMAPE for DeepState: {score:.3f}")
