from dataclasses import dataclass
from typing import Optional
from typing import Sequence
from typing import Union

CategoricalChoiceType = Union[None, bool, int, float, str]


@dataclass(frozen=True)
class BaseDistribution:
    """Base distribution."""

    def __post_init__(self):
        if self.__class__ == BaseDistribution:
            raise TypeError("Cannot instantiate abstract class.")


@dataclass(frozen=True)
class CategoricalDistribution(BaseDistribution):
    """Categorical distribution.

    The input parameters aren't validated.

    Look at :py:meth:`~optuna.trial.Trial.suggest_categorical` to find more about the meaning of parameters.

    Attributes
    ----------
    choices:
        Possible values to take.
    """

    choices: Sequence[CategoricalChoiceType]


@dataclass(frozen=True)
class IntDistribution(BaseDistribution):
    """Integer-based distribution.

    The input parameters aren't validated.

    Look at :py:meth:`~optuna.trial.Trial.suggest_int` to find more about the meaning of parameters.

    Attributes
    ----------
    low:
        The lower bound.
    high:
        The upper bound.
    step:
        The space between possible values.
    log:
        The flag of using log domain.
    """

    low: int
    high: int
    step: int = 1
    log: bool = False


@dataclass(frozen=True)
class FloatDistribution(BaseDistribution):
    """Float-based distribution.

    Look at :py:meth:`~optuna.trial.Trial.suggest_float` to find more about the meaning of parameters.

    Attributes
    ----------
    low:
        The lower bound.
    high:
        The upper bound.
    step:
        The space between possible values.
    log:
        The flag of using log domain.
    """

    low: float
    high: float
    step: Optional[float] = None
    log: bool = False
