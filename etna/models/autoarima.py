import warnings

import pandas as pd
import pmdarima as pm
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

from etna.models.base import PredictionIntervalContextIgnorantAbstractModel
from etna.models.mixins import PerSegmentModelMixin
from etna.models.mixins import PredictionIntervalContextIgnorantModelMixin
from etna.models.sarimax import _SARIMAXBaseAdapter

warnings.filterwarnings(
    message="No frequency information was provided, so inferred frequency .* will be used",
    action="ignore",
    category=ValueWarning,
    module="statsmodels.tsa.base.tsa_model",
)


class _AutoARIMAAdapter(_SARIMAXBaseAdapter):
    """
    Class for holding auto arima model.

    Notes
    -----
    We use auto ARIMA [1] model from pmdarima package.

    .. `auto ARIMA: <https://alkaline-ml.com/pmdarima/>_`

    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Init auto ARIMA model with given params.

        Parameters
        ----------
        **kwargs:
            Training parameters for auto_arima from pmdarima package.
        """
        self.kwargs = kwargs
        super().__init__()

    def _get_fit_results(self, endog: pd.Series, exog: pd.DataFrame) -> SARIMAXResultsWrapper:
        endog_np = endog.values
        model = pm.auto_arima(endog_np, X=exog, **self.kwargs)
        return model.arima_res_


class AutoARIMAModel(
    PerSegmentModelMixin, PredictionIntervalContextIgnorantModelMixin, PredictionIntervalContextIgnorantAbstractModel
):
    """
    Class for holding auto arima model.

    Method ``predict`` can use true target values only on train data on future data autoregression
    forecasting will be made even if targets are known.

    Notes
    -----
    We use :py:class:`pmdarima.arima.arima.ARIMA`.
    """

    def __init__(
        self,
        **kwargs,
    ):
        """
        Init auto ARIMA model with given params.

        Parameters
        ----------
        **kwargs:
            Training parameters for auto_arima from pmdarima package.
        """
        self.kwargs = kwargs
        super(AutoARIMAModel, self).__init__(
            base_model=_AutoARIMAAdapter(
                **self.kwargs,
            )
        )
