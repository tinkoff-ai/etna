from typing import Optional

import pandas as pd
from pytorch_lightning import Trainer

from etna.datasets import TSDataset
from etna.pipeline.base import BasePipeline


class NNPipeline(BasePipeline):
    def __init__(
        self,
        horizon: int,
        encoder_length: int,
        decoder_length: int,
        model,
        transforms,
        columns_to_add,
        datetime_index: Optional[str] = None,
        train_batch_size=32,
        test_batch_size=64,
        trainer_kwargs=dict(),
    ):
        super().__init__(horizon)
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.model = model
        self.transforms = transforms
        self.columns_to_add = columns_to_add
        self.datetime_index = datetime_index
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.trainer_kwargs = trainer_kwargs

    def fit(self, ts: TSDataset) -> "NNPipeline":
        self.ts = ts
        self.ts.fit_transform(self.transforms)

        train_dataloader = self.ts.to_train_dataloader(
            encoder_length=self.encoder_length,
            decoder_length=self.decoder_length,
            columns_to_add=self.columns_to_add,
            datetime_index=self.datetime_index,
            batch_size=self.train_batch_size,
        )
        trainer = Trainer(**self.trainer_kwargs)
        trainer.fit(model=self.model, train_dataloader=train_dataloader)

        self.ts.inverse_transform()
        return self

    def _forecast(self) -> TSDataset:
        """Make predictions."""
        if self.ts is None:
            raise ValueError("Something went wrong, ts is None!")

        test_dataloader = self.ts.to_test_dataloader(
            encoder_length=self.encoder_length,
            decoder_length=self.decoder_length,
            columns_to_add=self.columns_to_add,
            datetime_index=self.datetime_index,
            batch_size=self.test_batch_size,
        )
        future_ts = self.ts.make_future(self.horizon)

        for batch in test_dataloader:
            segment = batch["segment"]
            predictions = self.model(batch)
            future_ts.df.loc[:, pd.IndexSlice[segment, "target"]] = predictions.detach().numpy()[0]

        future_ts.inverse_transform()
        return future_ts
