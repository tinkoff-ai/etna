import inspect
import warnings
from enum import Enum


class BaseMixin:
    """Base mixin for etna classes."""

    def __repr__(self):
        """Get default representation of etna object."""
        # TODO: add tests default behaviour for all registered objects
        args_str_representation = ""
        init_args = inspect.signature(self.__init__).parameters
        for arg, param in init_args.items():
            if param.kind == param.VAR_POSITIONAL:
                continue
            elif param.kind == param.VAR_KEYWORD:
                for arg_, value in self.__dict__[arg].items():
                    args_str_representation += f"{arg_} = {repr(value)}, "
            else:
                try:
                    value = self.__dict__[arg]
                except KeyError as e:
                    value = None
                    warnings.warn(f"You haven't set all parameters inside class __init__ method: {e}")
                args_str_representation += f"{arg} = {repr(value)}, "
        return f"{self.__class__.__name__}({args_str_representation})"


class StringEnumWithRepr(str, Enum):
    """Base class for str enums, that has alternative __repr__ method."""

    def __repr__(self):
        """Get string representation for enum string so that enum can be created from it."""
        return self.value.__repr__()
