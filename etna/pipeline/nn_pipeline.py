import pandas as pd
from pytorch_lightning import Trainer
from sklearn.pipeline import Pipeline

from etna.datasets import TSDataset
from etna.pipeline.base import BasePipeline


class NNPipeline(BasePipeline):
    def __init__(
        self,
        horizon: int,
        model,
        transforms,
        train_batch_size=16,
        test_batch_size=1,
        trainer_kwargs=dict(),
    ):
        super().__init__(horizon=horizon)
        self.model = model
        self.transforms = transforms
        self.trainer_kwargs = trainer_kwargs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

    def fit(self, ts: TSDataset) -> "Pipeline":
        self.ts = ts
        self.ts.fit_transform(self.transforms)

        train_dataloader = self.ts.to_train_dataloader(
            make_samples=self.model.make_samples,
            batch_size=self.train_batch_size,
        )
        trainer = Trainer(**self.trainer_kwargs)
        trainer.fit(model=self.model, train_dataloaders=train_dataloader)

        self.ts.inverse_transform()
        return self

    def _forecast(self) -> TSDataset:
        """Make predictions."""
        if self.ts is None:
            raise ValueError("Something went wrong, ts is None!")

        test_dataloader = self.ts.to_test_dataloader(
            encoder_length=self.model.encoder_length,
            decoder_length=self.model.decoder_length,
            batch_size=self.test_batch_size,
            make_samples=self.model.make_samples,
        )
        future_ts = self.ts.make_future(self.horizon)

        for batch in test_dataloader:
            segment = batch["segment"]
            predictions = self.model(batch)
            future_ts.df.loc[:, pd.IndexSlice[segment, "target"]] = predictions.detach().numpy()[0]

        future_ts.inverse_transform()
        return future_ts
