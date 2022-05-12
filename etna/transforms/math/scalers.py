from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from etna.transforms.math.sklearn import SklearnTransform
from etna.transforms.math.sklearn import TransformMode


class StandardScalerTransform(SklearnTransform):
    """Standardize features by removing the mean and scaling to unit variance.

    Uses :py:class:`sklearn.preprocessing.StandardScaler` inside.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: Optional[Union[str, List[str]]] = None,
        inplace: bool = True,
        out_column: Optional[str] = None,
        with_mean: bool = True,
        with_std: bool = True,
        mode: Union[TransformMode, str] = "per-segment",
    ):
        """
        Init StandardScalerPreprocess.

        Parameters
        ----------
        in_column:
            columns to be scaled, if None - all columns will be scaled.
        inplace:
            features are changed by scaled.
        out_column:
            base for the names of generated columns, uses ``self.__repr__()`` if not given.
        with_mean:
            if True, center the data before scaling.
        with_std:
            if True, scale the data to unit standard deviation.
        mode:
            "macro" or "per-segment", way to transform features over segments.

            * If "macro", transforms features globally, gluing the corresponding ones for all segments.

            * If "per-segment", transforms features for each segment separately.

        Raises
        ------
        ValueError:
            if incorrect mode given
        """
        self.with_mean = with_mean
        self.with_std = with_std
        super().__init__(
            in_column=in_column,
            transformer=StandardScaler(with_mean=self.with_mean, with_std=self.with_std, copy=True),
            out_column=out_column,
            inplace=inplace,
            mode=mode,
        )


class RobustScalerTransform(SklearnTransform):
    """Scale features using statistics that are robust to outliers.

    Uses :py:class:`sklearn.preprocessing.RobustScaler` inside.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: Optional[Union[str, List[str]]] = None,
        inplace: bool = True,
        out_column: Optional[str] = None,
        with_centering: bool = True,
        with_scaling: bool = True,
        quantile_range: Tuple[float, float] = (25, 75),
        unit_variance: bool = False,
        mode: Union[TransformMode, str] = "per-segment",
    ):
        """
        Init RobustScalerPreprocess.

        Parameters
        ----------
        in_column:
            columns to be scaled, if None - all columns will be scaled.
        inplace:
            features are changed by scaled.
        out_column:
            base for the names of generated columns, uses ``self.__repr__()`` if not given.
        with_centering:
            if True, center the data before scaling.
        with_scaling:
            if True, scale the data to interquartile range.
        quantile_range:
            quantile range.
        unit_variance:
            If True, scale data so that normally distributed features have a variance of 1.

            In general, if the difference between the x-values of q_max and q_min for a standard normal
            distribution is greater than 1, the dataset will be scaled down. If less than 1,
            the dataset will be scaled up.
        mode:
            "macro" or "per-segment", way to transform features over segments.

            * If "macro", transforms features globally, gluing the corresponding ones for all segments.

            * If "per-segment", transforms features for each segment separately.

        Raises
        ------
        ValueError:
            if incorrect mode given
        """
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.unit_variance = unit_variance
        super().__init__(
            in_column=in_column,
            inplace=inplace,
            out_column=out_column,
            transformer=RobustScaler(
                with_centering=self.with_centering,
                with_scaling=self.with_scaling,
                quantile_range=self.quantile_range,
                unit_variance=self.unit_variance,
                copy=True,
            ),
            mode=mode,
        )


class MinMaxScalerTransform(SklearnTransform):
    """Transform features by scaling each feature to a given range.

    Uses :py:class:`sklearn.preprocessing.MinMaxScaler` inside.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: Optional[Union[str, List[str]]] = None,
        inplace: bool = True,
        out_column: Optional[str] = None,
        feature_range: Tuple[float, float] = (0, 1),
        clip: bool = True,
        mode: Union[TransformMode, str] = "per-segment",
    ):
        """
        Init MinMaxScalerPreprocess.

        Parameters
        ----------
        in_column:
            columns to be scaled, if None - all columns will be scaled.
        inplace:
            features are changed by scaled.
        out_column:
            base for the names of generated columns, uses ``self.__repr__()`` if not given.
        feature_range:
            desired range of transformed data.
        clip:
            set to True to clip transformed values of held-out data to provided feature range.
        mode:
            "macro" or "per-segment", way to transform features over segments.

            * If "macro", transforms features globally, gluing the corresponding ones for all segments.

            * If "per-segment", transforms features for each segment separately.

        Raises
        ------
        ValueError:
            if incorrect mode given
        """
        self.feature_range = feature_range
        self.clip = clip
        super().__init__(
            in_column=in_column,
            inplace=inplace,
            out_column=out_column,
            transformer=MinMaxScaler(feature_range=self.feature_range, clip=self.clip, copy=True),
            mode=mode,
        )


class MaxAbsScalerTransform(SklearnTransform):
    """Scale each feature by its maximum absolute value.

    Uses :py:class:`sklearn.preprocessing.MaxAbsScaler` inside.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: Optional[Union[str, List[str]]] = None,
        inplace: bool = True,
        out_column: Optional[str] = None,
        mode: Union[TransformMode, str] = "per-segment",
    ):
        """Init MinMaxScalerPreprocess.

        Parameters
        ----------
        in_column:
            columns to be scaled, if None - all columns will be scaled.
        inplace:
            features are changed by scaled.
        out_column:
            base for the names of generated columns, uses ``self.__repr__()`` if not given.
        mode:
            "macro" or "per-segment", way to transform features over segments.

            * If "macro", transforms features globally, gluing the corresponding ones for all segments.

            * If "per-segment", transforms features for each segment separately.

        Raises
        ------
        ValueError:
            if incorrect mode given
        """
        super().__init__(
            in_column=in_column,
            inplace=inplace,
            out_column=out_column,
            transformer=MaxAbsScaler(copy=True),
            mode=mode,
        )


__all__ = ["MaxAbsScalerTransform", "MinMaxScalerTransform", "RobustScalerTransform", "StandardScalerTransform"]
