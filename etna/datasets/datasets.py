import os
from pathlib import Path
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Union

import pandas as pd
import requests

from etna.datasets.tsdataset import TSDataset


class AirPassangersDataset:
    """Holds and loads air passanger dataset."""

    def __init__(self) -> None:
        self.name = "air_passangers.csv"
        self.default_path = Path(os.path.join(Path.home(), Path(".etna/datasets/air_passangers")))
        self.uri = "https://raw.githubusercontent.com/unit8co/darts/cf6364aadcd545131998b5e7099060b2432376e4/datasets/AirPassengers.csv"
        self.freq = "MS"
        self.known_future: Union[Literal["all"], Sequence] = ()

    def load(self) -> TSDataset:
        """Create TSDataset with air passangers data (loads if not present)."""
        if not self._is_already_downloaded():
            os.makedirs(self.default_path, exist_ok=True)
            try:
                req = requests.get(self.uri)
                with open(self.default_path / self.name, "wb") as f:
                    f.write(req.content)
            except requests.exceptions.RequestException as e:
                raise e

        df, df_exog = self._transform_dataset()

        return TSDataset(df, self.freq, df_exog, self.known_future)

    def _is_already_downloaded(self):
        if (self.default_path / self.name).exists():
            return True
        return False

    def _transform_dataset(self) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        df = pd.read_csv(self.default_path / self.name)
        df.rename(columns={"Month": "timestamp", "#Passengers": "target"}, inplace=True)
        df["segment"] = "main"
        return TSDataset.to_dataset(df), None
