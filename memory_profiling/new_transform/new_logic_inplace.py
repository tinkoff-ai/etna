from etna.transforms import MedianOutliersTransform, LinearTrendTransform, FilterFeaturesTransform, DateFlagsTransform, LagTransform, LabelEncoderTransform, TimeSeriesImputerTransform
from new_transform import NewTransformInplace
from etna.datasets import TSDataset
from etna.datasets import generate_ar_df

df = generate_ar_df(periods=10000, n_segments=10, freq="D", start_time="2000-01-01")
df = TSDataset.to_dataset(df)
ts = TSDataset(df=df, freq="D")

old_transforms = [
    MedianOutliersTransform(in_column="target"),
    TimeSeriesImputerTransform(in_column="target"),
    LinearTrendTransform(in_column="target"),
    DateFlagsTransform(True, True, True, True, True, True, True,True, True, out_column="dt"),
    LabelEncoderTransform(in_column="dt_day_number_in_week"),
    LagTransform(in_column="target", lags=100, out_column="lag"),
    FilterFeaturesTransform(exclude=["lag_1"])
]
new_transforms = [NewTransformInplace(transform) for transform in old_transforms]

for transform in new_transforms:
    ts = transform.fit_transform(ts)
