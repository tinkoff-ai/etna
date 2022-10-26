from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from catboost import Pool
from deprecated import deprecated

from etna.models.base import BaseAdapter
from etna.models.base import NonPredictionIntervalContextIgnorantAbstractModel
from etna.models.mixins import MultiSegmentModelMixin
from etna.models.mixins import NonPredictionIntervalContextIgnorantModelMixin
from etna.models.mixins import PerSegmentModelMixin


class _CatBoostAdapter(BaseAdapter):
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

    def _prepare_float_category_columns(self, df: pd.DataFrame):
        df[self._float_category_columns] = df[self._float_category_columns].astype(str).astype("category")

    def fit(self, df: pd.DataFrame, regressors: List[str]) -> "_CatBoostAdapter":
        """
        Fit Catboost model.

        Parameters
        ----------
        df:
            Features dataframe
        regressors:
            List of the columns with regressors(ignored in this model)

        Returns
        -------
        :
            Fitted model
        """
        features = df.drop(columns=["timestamp", "target"])
        target = df["target"]
        columns_dtypes = features.dtypes
        category_columns_dtypes = columns_dtypes[columns_dtypes == "category"]
        self._categorical = category_columns_dtypes.index.tolist()

        # select only columns with float categories
        float_category_columns_dtypes_indices = [
            idx
            for idx, x in enumerate(category_columns_dtypes)
            if issubclass(x.categories.dtype.type, (float, np.floating))
        ]
        float_category_columns_dtypes = category_columns_dtypes.iloc[float_category_columns_dtypes_indices]
        float_category_columns = float_category_columns_dtypes.index
        self._float_category_columns = float_category_columns
        self._prepare_float_category_columns(features)

        train_pool = Pool(features, target.values, cat_features=self._categorical)
        self.model.fit(train_pool)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute predictions from a Catboost model.

        Parameters
        ----------
        df:
            Features dataframe

        Returns
        -------
        :
            Array with predictions
        """
        features = df.drop(columns=["timestamp", "target"])
        self._prepare_float_category_columns(features)
        predict_pool = Pool(features, cat_features=self._categorical)
        pred = self.model.predict(predict_pool)
        return pred

    def get_model(self) -> CatBoostRegressor:
        """Get internal catboost.CatBoostRegressor model that is used inside etna class.

        Returns
        -------
        result:
           Internal model
        """
        return self.model


class CatBoostPerSegmentModel(
    PerSegmentModelMixin,
    NonPredictionIntervalContextIgnorantModelMixin,
    NonPredictionIntervalContextIgnorantAbstractModel,
):
    """Class for holding per segment Catboost model.

    Examples
    --------
    >>> from etna.datasets import generate_periodic_df
    >>> from etna.datasets import TSDataset
    >>> from etna.models import CatBoostPerSegmentModel
    >>> from etna.transforms import LagTransform
    >>> classic_df = generate_periodic_df(
    ...     periods=100,
    ...     start_time="2020-01-01",
    ...     n_segments=4,
    ...     period=7,
    ...     sigma=3
    ... )
    >>> df = TSDataset.to_dataset(df=classic_df)
    >>> ts = TSDataset(df, freq="D")
    >>> horizon = 7
    >>> transforms = [
    ...     LagTransform(in_column="target", lags=[horizon, horizon+1, horizon+2])
    ... ]
    >>> ts.fit_transform(transforms=transforms)
    >>> future = ts.make_future(horizon)
    >>> model = CatBoostPerSegmentModel()
    >>> model.fit(ts=ts)
    CatBoostPerSegmentModel(iterations = None, depth = None, learning_rate = None,
    logging_level = 'Silent', l2_leaf_reg = None, thread_count = None, )
    >>> forecast = model.forecast(future)
    >>> pd.options.display.float_format = '{:,.2f}'.format
    >>> forecast[:, :, "target"]
    segment    segment_0 segment_1 segment_2 segment_3
    feature       target    target    target    target
    timestamp
    2020-04-10      9.00      9.00      4.00      6.00
    2020-04-11      5.00      2.00      7.00      9.00
    2020-04-12      0.00      4.00      7.00      9.00
    2020-04-13      0.00      5.00      9.00      7.00
    2020-04-14      1.00      2.00      1.00      6.00
    2020-04-15      5.00      7.00      4.00      7.00
    2020-04-16      8.00      6.00      2.00      0.00
    """

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
        """Create instance of CatBoostPerSegmentModel with given parameters.

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

            * CPU — Any integer up to 16.

            * GPU — Any integer up to 8 pairwise modes (YetiRank, PairLogitPairwise and
              QueryCrossEntropy) and up to 16 for all other loss functions.
        learning_rate:
            The learning rate. Used for reducing the gradient step.
            If None the value is defined automatically depending on the number of iterations.
        logging_level:
            The logging level to output to stdout.
            Possible values:

            * Silent — Do not output any logging information to stdout.

            * Verbose — Output the following data to stdout:

                * optimized metric

                * elapsed time of training

                * remaining time of training

            * Info — Output additional information and the number of trees.

            * Debug — Output debugging information.

        l2_leaf_reg:
            Coefficient at the L2 regularization term of the cost function.
            Any positive value is allowed.
        thread_count:
            The number of threads to use during the training.

            * For CPU. Optimizes the speed of execution. This parameter doesn't affect results.
            * For GPU. The given value is used for reading the data from the hard drive and does
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
        super().__init__(
            base_model=_CatBoostAdapter(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                logging_level=logging_level,
                thread_count=thread_count,
                l2_leaf_reg=l2_leaf_reg,
                **kwargs,
            )
        )


class CatBoostMultiSegmentModel(
    MultiSegmentModelMixin,
    NonPredictionIntervalContextIgnorantModelMixin,
    NonPredictionIntervalContextIgnorantAbstractModel,
):
    """Class for holding Catboost model for all segments.

    Examples
    --------
    >>> from etna.datasets import generate_periodic_df
    >>> from etna.datasets import TSDataset
    >>> from etna.models import CatBoostMultiSegmentModel
    >>> from etna.transforms import LagTransform
    >>> classic_df = generate_periodic_df(
    ...     periods=100,
    ...     start_time="2020-01-01",
    ...     n_segments=4,
    ...     period=7,
    ...     sigma=3
    ... )
    >>> df = TSDataset.to_dataset(df=classic_df)
    >>> ts = TSDataset(df, freq="D")
    >>> horizon = 7
    >>> transforms = [
    ...     LagTransform(in_column="target", lags=[horizon, horizon+1, horizon+2])
    ... ]
    >>> ts.fit_transform(transforms=transforms)
    >>> future = ts.make_future(horizon)
    >>> model = CatBoostMultiSegmentModel()
    >>> model.fit(ts=ts)
    CatBoostMultiSegmentModel(iterations = None, depth = None, learning_rate = None,
    logging_level = 'Silent', l2_leaf_reg = None, thread_count = None, )
    >>> forecast = model.forecast(future)
    >>> pd.options.display.float_format = '{:,.2f}'.format
    >>> forecast[:, :, "target"].round()
    segment    segment_0 segment_1 segment_2 segment_3
    feature       target    target    target    target
    timestamp
    2020-04-10      9.00      9.00      4.00      6.00
    2020-04-11      5.00      2.00      7.00      9.00
    2020-04-12     -0.00      4.00      7.00      9.00
    2020-04-13      0.00      5.00      9.00      7.00
    2020-04-14      1.00      2.00      1.00      6.00
    2020-04-15      5.00      7.00      4.00      7.00
    2020-04-16      8.00      6.00      2.00      0.00
    """

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
        """Create instance of CatBoostMultiSegmentModel with given parameters.

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

            * CPU — Any integer up to 16.

            * GPU — Any integer up to 8 pairwise modes (YetiRank, PairLogitPairwise and
              QueryCrossEntropy) and up to 16 for all other loss functions.
        learning_rate:
            The learning rate. Used for reducing the gradient step.
            If None the value is defined automatically depending on the number of iterations.
        logging_level:
            The logging level to output to stdout.
            Possible values:

            * Silent — Do not output any logging information to stdout.

            * Verbose — Output the following data to stdout:

                * optimized metric

                * elapsed time of training

                * remaining time of training

            * Info — Output additional information and the number of trees.

            * Debug — Output debugging information.

        l2_leaf_reg:
            Coefficient at the L2 regularization term of the cost function.
            Any positive value is allowed.
        thread_count:
            The number of threads to use during the training.

            * For CPU. Optimizes the speed of execution. This parameter doesn't affect results.
            * For GPU. The given value is used for reading the data from the hard drive and does
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
        super().__init__(
            base_model=_CatBoostAdapter(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                logging_level=logging_level,
                thread_count=thread_count,
                l2_leaf_reg=l2_leaf_reg,
                **kwargs,
            )
        )


@deprecated(
    reason="CatBoostModelPerSegment is deprecated; will be deleted in etna==2.0. Use CatBoostPerSegmentModel instead."
)
class CatBoostModelPerSegment(CatBoostPerSegmentModel):
    """Class for holding per segment Catboost model.

    Warnings
    --------
    CatBoostModelPerSegment is deprecated; will be deleted in etna==2.0.
    Use etna.models.CatBoostPerSegmentModel instead.

    Examples
    --------
    >>> from etna.datasets import generate_periodic_df
    >>> from etna.datasets import TSDataset
    >>> from etna.models import CatBoostModelPerSegment
    >>> from etna.transforms import LagTransform
    >>> classic_df = generate_periodic_df(
    ...     periods=100,
    ...     start_time="2020-01-01",
    ...     n_segments=4,
    ...     period=7,
    ...     sigma=3
    ... )
    >>> df = TSDataset.to_dataset(df=classic_df)
    >>> ts = TSDataset(df, freq="D")
    >>> horizon = 7
    >>> transforms = [
    ...     LagTransform(in_column="target", lags=[horizon, horizon+1, horizon+2])
    ... ]
    >>> ts.fit_transform(transforms=transforms)
    >>> future = ts.make_future(horizon)
    >>> model = CatBoostModelPerSegment()
    >>> model.fit(ts=ts)
    CatBoostModelPerSegment(iterations = None, depth = None, learning_rate = None,
    logging_level = 'Silent', l2_leaf_reg = None, thread_count = None, )
    >>> forecast = model.forecast(future)
    >>> pd.options.display.float_format = '{:,.2f}'.format
    >>> forecast[:, :, "target"]
    segment    segment_0 segment_1 segment_2 segment_3
    feature       target    target    target    target
    timestamp
    2020-04-10      9.00      9.00      4.00      6.00
    2020-04-11      5.00      2.00      7.00      9.00
    2020-04-12      0.00      4.00      7.00      9.00
    2020-04-13      0.00      5.00      9.00      7.00
    2020-04-14      1.00      2.00      1.00      6.00
    2020-04-15      5.00      7.00      4.00      7.00
    2020-04-16      8.00      6.00      2.00      0.00
    """

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

            * CPU — Any integer up to 16.

            * GPU — Any integer up to 8 pairwise modes (YetiRank, PairLogitPairwise and
              QueryCrossEntropy) and up to 16 for all other loss functions.
        learning_rate:
            The learning rate. Used for reducing the gradient step.
            If None the value is defined automatically depending on the number of iterations.
        logging_level:
            The logging level to output to stdout.
            Possible values:

            * Silent — Do not output any logging information to stdout.

            * Verbose — Output the following data to stdout:

                * optimized metric

                * elapsed time of training

                * remaining time of training

            * Info — Output additional information and the number of trees.

            * Debug — Output debugging information.

        l2_leaf_reg:
            Coefficient at the L2 regularization term of the cost function.
            Any positive value is allowed.
        thread_count:
            The number of threads to use during the training.

            * For CPU. Optimizes the speed of execution. This parameter doesn't affect results.
            * For GPU. The given value is used for reading the data from the hard drive and does
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
        super().__init__(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            logging_level=logging_level,
            thread_count=thread_count,
            l2_leaf_reg=l2_leaf_reg,
            **kwargs,
        )


@deprecated(
    reason="CatBoostModelMultiSegment is deprecated; will be deleted in etna==2.0. "
    "Use CatBoostMultiSegmentModel instead."
)
class CatBoostModelMultiSegment(CatBoostMultiSegmentModel):
    """Class for holding Catboost model for all segments.

    Warnings
    --------
    CatBoostModelMultiSegment is deprecated; will be deleted in etna==2.0.
    Use etna.models.CatBoostMultiSegmentModel instead.

    Examples
    --------
    >>> from etna.datasets import generate_periodic_df
    >>> from etna.datasets import TSDataset
    >>> from etna.models import CatBoostModelMultiSegment
    >>> from etna.transforms import LagTransform
    >>> classic_df = generate_periodic_df(
    ...     periods=100,
    ...     start_time="2020-01-01",
    ...     n_segments=4,
    ...     period=7,
    ...     sigma=3
    ... )
    >>> df = TSDataset.to_dataset(df=classic_df)
    >>> ts = TSDataset(df, freq="D")
    >>> horizon = 7
    >>> transforms = [
    ...     LagTransform(in_column="target", lags=[horizon, horizon+1, horizon+2])
    ... ]
    >>> ts.fit_transform(transforms=transforms)
    >>> future = ts.make_future(horizon)
    >>> model = CatBoostModelMultiSegment()
    >>> model.fit(ts=ts)
    CatBoostModelMultiSegment(iterations = None, depth = None, learning_rate = None,
    logging_level = 'Silent', l2_leaf_reg = None, thread_count = None, )
    >>> forecast = model.forecast(future)
    >>> pd.options.display.float_format = '{:,.2f}'.format
    >>> forecast[:, :, "target"].round()
    segment    segment_0 segment_1 segment_2 segment_3
    feature       target    target    target    target
    timestamp
    2020-04-10      9.00      9.00      4.00      6.00
    2020-04-11      5.00      2.00      7.00      9.00
    2020-04-12     -0.00      4.00      7.00      9.00
    2020-04-13      0.00      5.00      9.00      7.00
    2020-04-14      1.00      2.00      1.00      6.00
    2020-04-15      5.00      7.00      4.00      7.00
    2020-04-16      8.00      6.00      2.00      0.00
    """

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

            * CPU — Any integer up to 16.

            * GPU — Any integer up to 8 pairwise modes (YetiRank, PairLogitPairwise and
              QueryCrossEntropy) and up to 16 for all other loss functions.
        learning_rate:
            The learning rate. Used for reducing the gradient step.
            If None the value is defined automatically depending on the number of iterations.
        logging_level:
            The logging level to output to stdout.
            Possible values:

            * Silent — Do not output any logging information to stdout.

            * Verbose — Output the following data to stdout:

                * optimized metric

                * elapsed time of training

                * remaining time of training

            * Info — Output additional information and the number of trees.

            * Debug — Output debugging information.

        l2_leaf_reg:
            Coefficient at the L2 regularization term of the cost function.
            Any positive value is allowed.
        thread_count:
            The number of threads to use during the training.

            * For CPU. Optimizes the speed of execution. This parameter doesn't affect results.
            * For GPU. The given value is used for reading the data from the hard drive and does
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
        super().__init__(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            logging_level=logging_level,
            thread_count=thread_count,
            l2_leaf_reg=l2_leaf_reg,
            **kwargs,
        )
