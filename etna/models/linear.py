import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression

from etna.models.base import NonPredictionIntervalContextIgnorantAbstractModel
from etna.models.mixins import MultiSegmentModelMixin
from etna.models.mixins import NonPredictionIntervalContextIgnorantModelMixin
from etna.models.mixins import PerSegmentModelMixin
from etna.models.sklearn import _SklearnAdapter


class _LinearAdapter(_SklearnAdapter):
    def predict_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate prediction components.

        Parameters
        ----------
        df:
            features dataframe

        Returns
        -------
        :
            dataframe with prediction components
        """
        if self.regressor_columns is None:
            raise ValueError("Model is not fitted! Fit the model before estimating forecast components!")

        components_coefs = self.model.coef_
        target_components = df[self.model.feature_names_in_].apply(pd.to_numeric)
        target_components = components_coefs * target_components
        if self.model.fit_intercept:
            target_components["intercept"] = self.model.intercept_
        target_components = target_components.add_prefix("target_component_")
        return target_components


class LinearPerSegmentModel(
    PerSegmentModelMixin,
    NonPredictionIntervalContextIgnorantModelMixin,
    NonPredictionIntervalContextIgnorantAbstractModel,
):
    """
    Class holding per segment :py:class:`sklearn.linear_model.LinearRegression`.

    Notes
    -----
    Target components are formed as the terms from linear regression formula.
    """

    def __init__(self, fit_intercept: bool = True, **kwargs):
        """
        Create instance of LinearModel with given parameters.

        Parameters
        ----------
        fit_intercept:
            Whether to calculate the intercept for this model. If set to False, no intercept will be used in
            calculations (i.e. data is expected to be centered).
        """
        self.fit_intercept = fit_intercept
        self.kwargs = kwargs
        super().__init__(
            base_model=_LinearAdapter(regressor=LinearRegression(fit_intercept=self.fit_intercept, **self.kwargs))
        )


class ElasticPerSegmentModel(
    PerSegmentModelMixin,
    NonPredictionIntervalContextIgnorantModelMixin,
    NonPredictionIntervalContextIgnorantAbstractModel,
):
    """
    Class holding per segment :py:class:`sklearn.linear_model.ElasticNet`.

    Notes
    -----
    Target components are formed as the terms from linear regression formula.
    """

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, fit_intercept: bool = True, **kwargs):
        """
        Create instance of ElasticNet with given parameters.

        Parameters
        ----------
        alpha:
            Constant that multiplies the penalty terms. Defaults to 1.0.
            ``alpha = 0`` is equivalent to an ordinary least square, solved by the LinearRegression object.
            For numerical reasons, using ``alpha = 0`` with the Lasso object is not advised.
            Given this, you should use the :py:class:`~etna.models.linear.LinearPerSegmentModel` object.
        l1_ratio:
            The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``.

            * For ``l1_ratio = 0`` the penalty is an L2 penalty.

            * For ``l1_ratio = 1`` it is an L1 penalty.

            * For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.

        fit_intercept:
            Whether to calculate the intercept for this model. If set to False, no intercept will be used in
            calculations (i.e. data is expected to be centered).
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.kwargs = kwargs
        super().__init__(
            base_model=_LinearAdapter(
                regressor=ElasticNet(
                    alpha=self.alpha,
                    l1_ratio=self.l1_ratio,
                    fit_intercept=self.fit_intercept,
                    **self.kwargs,
                )
            )
        )


class LinearMultiSegmentModel(
    MultiSegmentModelMixin,
    NonPredictionIntervalContextIgnorantModelMixin,
    NonPredictionIntervalContextIgnorantAbstractModel,
):
    """
    Class holding :py:class:`sklearn.linear_model.LinearRegression` for all segments.

    Notes
    -----
    Target components are formed as the terms from linear regression formula.
    """

    def __init__(self, fit_intercept: bool = True, **kwargs):
        """
        Create instance of LinearModel with given parameters.

        Parameters
        ----------
        fit_intercept:
            Whether to calculate the intercept for this model. If set to False, no intercept will be used in
            calculations (i.e. data is expected to be centered).
        """
        self.fit_intercept = fit_intercept
        self.kwargs = kwargs
        super().__init__(
            base_model=_LinearAdapter(regressor=LinearRegression(fit_intercept=self.fit_intercept, **self.kwargs))
        )


class ElasticMultiSegmentModel(
    MultiSegmentModelMixin,
    NonPredictionIntervalContextIgnorantModelMixin,
    NonPredictionIntervalContextIgnorantAbstractModel,
):
    """
    Class holding :py:class:`sklearn.linear_model.ElasticNet` for all segments.

    Notes
    -----
    Target components are formed as the terms from linear regression formula.
    """

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5, fit_intercept: bool = True, **kwargs):
        """
        Create instance of ElasticNet with given parameters.

        Parameters
        ----------
        alpha:
            Constant that multiplies the penalty terms. Defaults to 1.0.
            ``alpha = 0`` is equivalent to an ordinary least square, solved by the LinearRegression object.
            For numerical reasons, using ``alpha = 0`` with the Lasso object is not advised.
            Given this, you should use the :py:class:`~etna.models.linear.LinearMultiSegmentModel` object.
        l1_ratio:
            The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``.

            * For ``l1_ratio = 0`` the penalty is an L2 penalty.

            * For ``l1_ratio = 1`` it is an L1 penalty.

            * For ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2.

        fit_intercept:
            Whether to calculate the intercept for this model. If set to False, no intercept will be used in
            calculations (i.e. data is expected to be centered).
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.kwargs = kwargs
        super().__init__(
            base_model=_LinearAdapter(
                regressor=ElasticNet(
                    alpha=self.alpha,
                    l1_ratio=self.l1_ratio,
                    fit_intercept=self.fit_intercept,
                    **self.kwargs,
                )
            )
        )
