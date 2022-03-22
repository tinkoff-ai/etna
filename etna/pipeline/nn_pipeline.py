import pandas as pd
from pytorch_lightning import Trainer
from sklearn.pipeline import Pipeline

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
        trainer_kwargs=dict(),
    ):
        super().__init__(horizon)
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.model = model
        self.transforms = transforms
        self.columns_to_add = columns_to_add
        self.trainer_kwargs = trainer_kwargs

    def fit(self, ts: TSDataset) -> "Pipeline":
        self.ts = ts
        self.ts.fit_transform(self.transforms)

        train_dataloader = self.ts.to_train_dataloader(
            encoder_length=self.encoder_length,
            decoder_length=self.decoder_length,
            columns_to_add=self.columns_to_add,
            batch_size=3,
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
            batch_size=1,
        )
        future_ts = self.ts.make_future(self.horizon)

        for batch in test_dataloader:
            segment = batch["segment"]
            predictions = self.model(batch)
            future_ts.df.loc[:, pd.IndexSlice[segment, "target"]] = predictions.detach().numpy()[0]

        future_ts.inverse_transform()
        return future_ts
