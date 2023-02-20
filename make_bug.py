import pandas as pd
import numpy as np

from etna.datasets.tsdataset import TSDataset
from etna.pipeline import Pipeline
from etna.transforms import DateFlagsTransform
from etna.transforms import LagTransform
from etna.transforms import PytorchForecastingTransform
from pytorch_forecasting.data import GroupNormalizer
from etna.models.nn import TFTModel


original_df = pd.DataFrame(np.array([["2021-05-31", 1, 3],
                                     ["2021-06-07", 1, 6],
                                     ["2021-06-14", 1, 9],
                                     ["2021-06-21", 1, 12],
                                     ["2021-06-28", 1, 15]]),
                           columns=['timestamp', 'segment', 'target'])
original_df['timestamp'] = pd.to_datetime(original_df['timestamp'])
original_df['target'] = original_df['target'].astype(float)
df = TSDataset.to_dataset(original_df)
ts = TSDataset(df, freq="W-MON")

HORIZON = 1
transform_date = DateFlagsTransform(day_number_in_week=True, day_number_in_month=False, out_column="dateflag")
num_lags = 2
transform_lag = LagTransform(
    in_column="target",
    lags=[HORIZON + i for i in range(num_lags)],
    out_column="target_lag",
)

transform_tft = PytorchForecastingTransform(
    max_encoder_length=HORIZON,
    max_prediction_length=HORIZON,
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["target"],
    time_varying_known_categoricals=["dateflag_day_number_in_week"],
    static_categoricals=["segment"],
    target_normalizer=GroupNormalizer(groups=["segment"]),
)

model_tft = TFTModel(max_epochs=5, learning_rate=[0.1], gpus=0, batch_size=64)

pipeline_tft = Pipeline(
    model=model_tft,
    horizon=HORIZON,
    transforms=[transform_lag, transform_date, transform_tft],
)

pipeline_tft.save("666")
pipeline_tft.load("666")
pipeline_tft.fit(ts)
pipeline_tft.save("666")
pipeline_tft.load("666")
