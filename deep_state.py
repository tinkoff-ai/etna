# %%
import pandas as pd
import numpy as np
import torch
import random

from etna.models.nn.rnn import RNN
from etna.pipeline import Pipeline
from etna.analysis import plot_backtest
from etna.metrics import MAE, SMAPE, MSE
from etna.datasets import TSDataset
from etna.transforms import StandardScalerTransform, LagTransform, SegmentEncoderTransform

import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

from etna.datasets import TSDataset, generate_ar_df
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.distributions.normal import Normal
from etna.models.nn.deepstate import LevelSSM, LevelTrendSSM, SeasonalitySSM
from etna.pipeline import Pipeline
from etna.analysis import plot_forecast, plot_backtest
from etna.metrics import SMAPE
from etna.models.nn.deepstate import DeepStateNetwork

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# %%
original_df = pd.read_csv("examples/data/example_dataset.csv")
original_df.head()

# %%
df = TSDataset.to_dataset(original_df)
ts = TSDataset(df, freq="D")
ts.head(5)

# %%

n_groups = 5
n_samples = 140
seq_length = 42
horizon = 14
encoder_length = 28
ssm = SeasonalitySSM(num_seasons=7)

# %%
pipe = Pipeline(
        horizon=horizon,
        model=DeepStateNetwork(
            ssm, encoder_length=encoder_length, decoder_length=0, n_samples=n_samples, input_size=1,
            train_batch_size=32, test_batch_size=1,
            trainer_kwargs=dict(max_epochs=20),
        ),
        transforms=[SegmentEncoderTransform()],
    )

# %%
metrics, forecast, fold_info = pipe.backtest(
    ts, metrics=[SMAPE(), MAE(), MSE()],
    n_folds=3, n_jobs=1
)

# %%
plot_backtest(forecast, ts, history_len=20)

# %%
score = metrics["SMAPE"].mean()
print(f"Average SMAPE for LSTM: {score:.3f}")

# %%



