from typing import Dict

from etna import SETTINGS
from etna.models.seasonal_ma import SeasonalMovingAverageModel

if SETTINGS.auto_required:
    from optuna.distributions import BaseDistribution


class NaiveModel(SeasonalMovingAverageModel):
    """Naive model predicts t-th value of series with its (t - lag) value.

    .. math::
        y_{t} = y_{t-s},

    where :math:`s` is lag.
    """

    def __init__(self, lag: int = 1):
        """
        Init NaiveModel.

        Parameters
        ----------
        lag: int
            lag for new value prediction
        """
        self.lag = lag
        super().__init__(window=1, seasonality=lag)

    def params_to_tune(self) -> Dict[str, "BaseDistribution"]:
        """Get default grid for tuning hyperparameters.

        This grid is empty.

        Returns
        -------
        :
            Grid to tune.
        """
        return {}


__all__ = ["NaiveModel"]
