import os
from dataclasses import dataclass

import pandas as pd

from etna import MODULE


@dataclass()
class Dataset:
    dataset_url: str
    freq: str
    dirname: str
    fname: str

    @property
    def cache_path(self):
        return os.path.join(MODULE, self.dirname, self.fname)


MonthlyWineSales = Dataset(
    dataset_url="https://raw.githubusercontent.com/tinkoff-ai/etna/master/examples/data/monthly-australian-wine-sales.csv",
    freq="D",
    dirname="monthly_australian_wine_sales",
    fname="monthly_australian_wine_sales.csv"
)


def load_dataset(self):
    print(self.cache_path)
    if os.path.exists(self.cache_path):
        return True
    os.mkdir(os.path.join(MODULE, self.DIRNAME))
    df = pd.read_csv(self.dataset_url)
    df.to_csv(self.cache_path)
    return True
