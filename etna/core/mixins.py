import inspect
import warnings


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
                    args_str_representation += f"{arg_} = {value.__repr__()}, "
            else:
                try:
                    value = self.__dict__[arg]
                except KeyError as e:
                    value = None
                    warnings.warn(f"You haven't set all parameters inside class __init__ method: {e}")
                args_str_representation += f"{arg} = {value.__repr__()}, "
        return f"{self.__class__.__name__}({args_str_representation})"

    def set_logger(self, logger):
        """Propаgate logger to all args of current object."""
        init_args = inspect.signature(self.__init__).parameters
        if "logger" in init_args:
            self.logger = logger
        for arg in init_args:
            value = self.__dict__.get(arg)
            if getattr(value, "set_logger", None) is not None:
                value.set_logger(logger)
