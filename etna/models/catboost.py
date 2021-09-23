from typing import Optional

import pandas as pd
from catboost import CatBoostRegressor
from catboost import Pool

from etna.datasets.tsdataset import TSDataset
from etna.models.base import Model
from etna.models.base import PerSegmentModel
from etna.models.base import log_decorator


class _CatBoostModel:
    def __init__(
        self,
        iterations: Optional[int] = None,
        depth: Optional[int] = None,
        learning_rate: Optional[float] = None,
        logging_level: Optional[str] = "Silent",
        l2_leaf_reg: Optional[float] = None,
        thread_count: Optional[int] = None,
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
        pred = self.model.predict(predict_pool)
        return pred


class CatBoostModelPerSegment(PerSegmentModel):
    """Class for holding per segment Catboost model."""

    def __init__(
        self,
        iterations: Optional[int] = None,
        depth: Optional[int] = None,
        learning_rate: Optional[float] = None,
        logging_level: Optional[str] = "Silent",
        l2_leaf_reg: Optional[float] = None,
        thread_count: Optional[int] = None,
        **kwargs,
    ):
        """Create instance of CatBoostModelPerSegment with given parameters.

        Parameters
        ----------
        iterations:
            The maximum number of trees that can be built when solving
            machine learning problems. When using other parameters that
            limit the number of iterations, the final number of trees
            may be less than the number specified in this parameter.
        depth:
            Depth of the tree. The range of supported values depends
            on the processing unit type and the type of the selected loss function:
            CPU — Any integer up to  16.
            GPU — Any integer up to 8 pairwise modes (YetiRank, PairLogitPairwise and
            QueryCrossEntropy) and up to   16 for all other loss functions.
        learning_rate:
            The learning rate. Used for reducing the gradient step.
            If None the value is defined automatically depending on the number of iterations.
        logging_level:
            The logging level to output to stdout.
            Possible values:
            Silent — Do not output any logging information to stdout.
            Verbose — Output the following data to stdout:
                optimized metric
                elapsed time of training
                remaining time of training
            Info — Output additional information and the number of trees.
            Debug — Output debugging information.
        l2_leaf_reg:
            Coefficient at the L2 regularization term of the cost function.
            Any positive value is allowed.
        thread_count:
            The number of threads to use during the training.
            For CPU
            Optimizes the speed of execution. This parameter doesn't affect results.
            For GPU
            The given value is used for reading the data from the hard drive and does
            not affect the training.
            During the training one main thread and one thread for each GPU are used.
        """
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.logging_level = logging_level
        self.l2_leaf_reg = l2_leaf_reg
        self.thread_count = thread_count
        self.kwargs = kwargs
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
        iterations: Optional[int] = None,
        depth: Optional[int] = None,
        learning_rate: Optional[float] = None,
        logging_level: Optional[str] = "Silent",
        l2_leaf_reg: Optional[float] = None,
        thread_count: Optional[int] = None,
        **kwargs,
    ):
        """Create instance of CatBoostModelMultiSegment with given parameters.

        Parameters
        ----------
        iterations:
            The maximum number of trees that can be built when solving
            machine learning problems. When using other parameters that
            limit the number of iterations, the final number of trees
            may be less than the number specified in this parameter.
        depth:
            Depth of the tree. The range of supported values depends
            on the processing unit type and the type of the selected loss function:
            CPU — Any integer up to  16.
            GPU — Any integer up to 8 pairwise modes (YetiRank, PairLogitPairwise and
            QueryCrossEntropy) and up to   16 for all other loss functions.
        learning_rate:
            The learning rate. Used for reducing the gradient step.
            If None the value is defined automatically depending on the number of iterations.
        logging_level:
            The logging level to output to stdout.
            Possible values:
            Silent — Do not output any logging information to stdout.
            Verbose — Output the following data to stdout:
                optimized metric
                elapsed time of training
                remaining time of training
            Info — Output additional information and the number of trees.
            Debug — Output debugging information.
        l2_leaf_reg:
            Coefficient at the L2 regularization term of the cost function.
            Any positive value is allowed.
        thread_count:
            The number of threads to use during the training.
            For CPU
            Optimizes the speed of execution. This parameter doesn't affect results.
            For GPU
            The given value is used for reading the data from the hard drive and does
            not affect the training.
            During the training one main thread and one thread for each GPU are used.
        """
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.logging_level = logging_level
        self.l2_leaf_reg = l2_leaf_reg
        self.thread_count = thread_count
        self.kwargs = kwargs
        super(CatBoostModelMultiSegment, self).__init__()
        self._base_model = _CatBoostModel(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            logging_level=logging_level,
            thread_count=thread_count,
            l2_leaf_reg=l2_leaf_reg,
            **kwargs,
        )

    @log_decorator
    def fit(self, ts: TSDataset) -> "CatBoostModelMultiSegment":
        """Fit model."""
        df = ts.to_pandas(flatten=True)
        df = df.dropna()
        df = df.drop(columns="segment")
        self._base_model.fit(df=df)
        return self

    @log_decorator
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
        result_list = list()
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
