from etna.datasets import TSDataset
from etna.transforms.base import Transform


class NewTransformCopy:
    """Decorator adding new transform behaviour."""

    def __init__(self, transform: Transform):
        self.transform = transform

    def fit_transform(self, ts: TSDataset):
        """Apply transform to the dataset creating new dataset."""
        new_df = ts.df.copy()
        new_df = self.transform.fit_transform(new_df)
        new_ts = TSDataset(df=new_df, freq=ts.freq)
        return new_ts


class NewTransformInplace:
    """Decorator adding new transform behaviour."""

    def __init__(self, transform: Transform):
        self.transform = transform

    def fit_transform(self, ts: TSDataset):
        """Apply transform to the dataset."""
        ts.df = self.transform.fit_transform(ts.df)
        return ts
