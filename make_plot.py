import matplotlib.pyplot as plt
import pandas as pd

from etna.analysis import plot_metric_per_segment
from etna.metrics import SMAPE, MAE
from etna.models import ProphetModel
from etna.pipeline import Pipeline
from etna.datasets import TSDataset


def main():
    df = pd.read_csv("examples/data/example_dataset.csv", parse_dates=["timestamp"])
    df_wide = TSDataset.to_dataset(df)
    ts = TSDataset(df=df_wide, freq="D")

    pipeline = Pipeline(model=ProphetModel(), horizon=10)
    metrics_df, _, _ = pipeline.backtest(ts=ts, n_folds=3, metrics=[MAE(), SMAPE()])

    plot_metric_per_segment(metrics_df=metrics_df, metric_name="MAE", per_fold_aggregation_mode="mean")
    plt.savefig("metric_per_segment_plot")

    pipeline = Pipeline(model=ProphetModel(), horizon=10)
    metrics_df, _, _ = pipeline.backtest(ts=ts, n_folds=3, metrics=[MAE(), SMAPE()])

    plot_metric_per_segment(metrics_df=metrics_df, metric_name="MAE", per_fold_aggregation_mode="mean", ascending=True, top_k=2)
    plt.savefig("metric_per_segment_plot_limited")


if __name__ == "__main__":
    main()
