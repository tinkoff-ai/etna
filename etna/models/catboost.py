import pandas as pd
from catboost import CatBoostRegressor
from catboost import Pool

from etna.datasets.tsdataset import TSDataset
from etna.models.base import Model
from etna.models.base import PerSegmentModel


class _CatBoostModel:
    def __init__(
        self,
        iterations: int = 100,
        depth: int = 4,
        learning_rate: float = 0.23,
        logging_level: str = "Silent",
        l2_leaf_reg: float = 6.735163225977638,
        thread_count: int = 4,
        **kwargs,
    ):
        self.model = CatBoostRegressor(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            logging_level=logging_level,
            thread_count=thread_count,
            l2_leaf_reg=l2_leaf_reg,
            **kwargs,
        )
        self._categorical = None

    def fit(self, df: pd.DataFrame) -> "_CatBoostModel":
        features = df.drop(columns=["timestamp", "target"])
        target = df["target"]
        self._categorical = features.select_dtypes(include=["category"]).columns.to_list()
        train_pool = Pool(features, target.values, cat_features=self._categorical)
        self.model.fit(train_pool)
        return self

    def predict(self, df: pd.DataFrame) -> list:
        features = df.drop(columns=["timestamp", "target"])
        predict_pool = Pool(features, cat_features=self._categorical)
        return self.model.predict(predict_pool)


class CatBoostModelPerSegment(PerSegmentModel):
    """Class for holding per segment Catboost model."""

    def __init__(
        self,
        iterations: int = 100,
        depth: int = 4,
        learning_rate: float = 0.23,
        logging_level: str = "Silent",
        l2_leaf_reg: float = 6.735163225977638,
        thread_count: int = 4,
        **kwargs,
    ):
        super(CatBoostModelPerSegment, self).__init__(
            base_model=_CatBoostModel(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                logging_level=logging_level,
                thread_count=thread_count,
                l2_leaf_reg=l2_leaf_reg,
                **kwargs,
            )
        )


class CatBoostModelMultiSegment(Model):
    """Class for holding Catboost model for all segments."""

    def __init__(
        self,
        iterations: int = 100,
        depth: int = 4,
        learning_rate: float = 0.23,
        logging_level: str = "Silent",
        l2_leaf_reg: float = 6.735163225977638,
        thread_count: int = 4,
        **kwargs,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.logging_level = logging_level
        self.l2_leaf_reg = l2_leaf_reg
        self.thread_count = thread_count
        self.kwargs = kwargs
        super(CatBoostModelMultiSegment, self).__init__()
        self._base_model = _CatBoostModel(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            logging_level=self.logging_level,
            thread_count=self.thread_count,
            l2_leaf_reg=self.l2_leaf_reg,
            **self.kwargs,
        )

    def fit(self, ts: TSDataset) -> "CatBoostModelMultiSegment":
        """Fit model."""
        df = ts.to_pandas(flatten=True)
        df = df.dropna()
        df = df.drop(columns="segment")
        self._base_model.fit(df=df)
        return self

    def forecast(self, ts: TSDataset) -> TSDataset:
        """Make predictions.

        Parameters
        ----------
        ts:
            Dataframe with features
        Returns
        -------
        DataFrame
            Models result
        """
        result_list = []
        for segment in ts.segments:
            segment_predict = self._forecast_segment(self._base_model, segment, ts)
            result_list.append(segment_predict)

        result_df = pd.concat(result_list, ignore_index=True)
        result_df = result_df.set_index(["timestamp", "segment"])

        df = ts.to_pandas(flatten=True)
        df = df.set_index(["timestamp", "segment"])
        df = df.combine_first(result_df).reset_index()

        df = TSDataset.to_dataset(df)
        ts.df = df
        ts.inverse_transform()
        return ts
