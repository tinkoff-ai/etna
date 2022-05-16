from typing import Optional

from ruptures.base import BaseCost
from ruptures.detection import Binseg
from sklearn.linear_model import LinearRegression

from etna.transforms.decomposition.change_points_trend import ChangePointsTrendTransform
from etna.transforms.decomposition.change_points_trend import TDetrendModel


class BinsegTrendTransform(ChangePointsTrendTransform):
    """BinsegTrendTransform uses :py:class:`ruptures.detection.Binseg` model as a change point detection model.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(
        self,
        in_column: str,
        detrend_model: Optional[TDetrendModel] = None,
        model: str = "ar",
        custom_cost: Optional[BaseCost] = None,
        min_size: int = 2,
        jump: int = 1,
        n_bkps: int = 5,
        pen: Optional[float] = None,
        epsilon: Optional[float] = None,
    ):
        """Init BinsegTrendTransform.

        Parameters
        ----------
        in_column:
            name of column to apply transform to
        detrend_model:
            model to get trend in data
        model:
            binseg segment model, ["l1", "l2", "rbf",...]. Not used if 'custom_cost' is not None.
        custom_cost:
            binseg custom cost function
        min_size:
            minimum segment length necessary to decide it is a stable trend segment
        jump:
            jump value can speed up computations: if ``jump==k``,
            the algo will use every k-th value for change points search.
        n_bkps:
            number of change points to find
        pen:
            penalty value (>0)
        epsilon:
            reconstruction budget (>0)
        """
        self.model = model
        self.custom_cost = custom_cost
        self.min_size = min_size
        self.jump = jump
        self.n_bkps = n_bkps
        self.pen = pen
        self.epsilon = epsilon
        detrend_model = LinearRegression() if detrend_model is None else detrend_model
        super().__init__(
            in_column=in_column,
            change_point_model=Binseg(
                model=self.model, custom_cost=self.custom_cost, min_size=self.min_size, jump=self.jump
            ),
            detrend_model=detrend_model,
            n_bkps=self.n_bkps,
            pen=self.pen,
            epsilon=self.epsilon,
        )
