from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import List
from typing import TypeVar
from typing import Union

T = TypeVar("T")


class AbstractRunner(ABC):
    """Abstract class for Runner.

    Note
    ----
    This class requires ``auto`` extension to be installed.
    Read more about this at :ref:`installation page <installation>`.
    """

    @abstractmethod
    def __call__(self, func: Callable[..., T], *args, **kwargs) -> Union[T, List[T]]:
        """Call given ``func`` in specified environment with ``*args`` and ``**kwargs``."""
        pass
