import inspect
from copy import deepcopy
from functools import wraps
from typing import Callable


def init_collector(init: Callable) -> Callable:
    """Make decorator for collecting init parameters."""

    @wraps(init)
    def wrapper(*args, **kwargs):
        self, *args = args
        init_args = inspect.signature(self.__init__).parameters

        deepcopy_args = deepcopy(args)
        deepcopy_kwargs = deepcopy(kwargs)

        self._init_params = {}
        args_dict = dict(
            zip([arg for arg, param in init_args.items() if param.kind == param.POSITIONAL_OR_KEYWORD], deepcopy_args)
        )
        self._init_params.update(args_dict)
        self._init_params.update(deepcopy_kwargs)

        return init(self, *args, **kwargs)

    return wrapper
