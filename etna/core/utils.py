import inspect
from copy import deepcopy
from functools import wraps
from typing import Callable


def init_collector(init: Callable) -> Callable:
    """
    Make decorator for collecting init parameters.
    N.B. if init method has positional only parameters, they will be ignored.
    """

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


def create_type_with_init_collector(type_: type) -> type:
    """Create type with init decorated with init_collector."""
    previous_frame = inspect.stack()[1]
    module = inspect.getmodule(previous_frame[0])
    if module is None:
        return type_
    new_type = type(type_.__name__, (type_,), {"__module__": module.__name__})
    if hasattr(type_, "__init__"):
        new_type.__init__ = init_collector(new_type.__init__)  # type: ignore
    return new_type
