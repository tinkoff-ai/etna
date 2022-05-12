from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression

from etna.models.sklearn import SklearnMultiSegmentModel
from etna.models.sklearn import SklearnPerSegmentModel


class LinearPerSegmentModel(SklearnPerSegmentModel):
    """Class holding per segment :py:class:`sklearn.linear_model.LinearRegression`."""

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
        super().__init__(regressor=LinearRegression(fit_intercept=self.fit_intercept, **self.kwargs))


class ElasticPerSegmentModel(SklearnPerSegmentModel):
    """Class holding per segment :py:class:`sklearn.linear_model.ElasticNet`."""

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
            regressor=ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept,
                **self.kwargs,
            )
        )


class LinearMultiSegmentModel(SklearnMultiSegmentModel):
    """Class holding :py:class:`sklearn.linear_model.LinearRegression` for all segments."""

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
        super().__init__(regressor=LinearRegression(fit_intercept=self.fit_intercept, **self.kwargs))


class ElasticMultiSegmentModel(SklearnMultiSegmentModel):
    """Class holding :py:class:`sklearn.linear_model.ElasticNet` for all segments."""

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
            regressor=ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept,
                **self.kwargs,
            )
        )
