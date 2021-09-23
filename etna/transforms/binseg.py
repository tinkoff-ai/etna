from functools import lru_cache
from typing import Any
from typing import Optional

from ruptures.base import BaseCost
from ruptures.costs import cost_factory
from ruptures.detection import Binseg
from sklearn.linear_model import LinearRegression

from etna.transforms.change_points_trend import ChangePointsTrendTransform
from etna.transforms.change_points_trend import TDetrendModel


class _Binseg(Binseg):
    """Binary segmentation with lru_cache."""

    def __init__(
        self,
        model: str = "l2",
        custom_cost: Optional[BaseCost] = None,
        min_size: int = 2,
        jump: int = 5,
        params: Any = None,
    ):
        """Initialize a Binseg instance.

        Args:
            model (str, optional): segment model, ["l1", "l2", "rbf",...]. Not used if ``'custom_cost'`` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length. Defaults to 2 samples.
            jump (int, optional): subsample (one every *jump* points). Defaults to 5 samples.
            params (dict, optional): a dictionary of parameters for the cost instance.
        """
        if custom_cost is not None and isinstance(custom_cost, BaseCost):
            self.cost = custom_cost
        elif params is None:
            self.cost = cost_factory(model=model)
        else:
            self.cost = cost_factory(model=model, **params)
        self.min_size = max(min_size, self.cost.min_size)
        self.jump = jump
        self.n_samples = None
        self.signal = None

    @lru_cache(maxsize=None)
    def single_bkp(self, start: int, end: int) -> Any:
        """Run _single_bkp with lru_cache decorator."""
        return self._single_bkp(start=start, end=end)


class BinsegTrendTransform(ChangePointsTrendTransform):
    """BinsegTrendTransform uses _Binseg model as a change point detection model in ChangePointsTrendTransform transform."""

    def __init__(
        self,
        in_column: str,
        detrend_model: TDetrendModel = LinearRegression(),
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
            jump value can speed up computations: if jump==k, the algo will use every k-th value for change points search.
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
        super().__init__(
            in_column=in_column,
            change_point_model=_Binseg(
                model=self.model, custom_cost=self.custom_cost, min_size=self.min_size, jump=self.jump
            ),
            detrend_model=detrend_model,
            n_bkps=self.n_bkps,
            pen=self.pen,
            epsilon=self.epsilon,
        )
