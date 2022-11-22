import functools
import inspect

from etna.loggers import tslogger


# TODO: make PyCharm see signature of decorated method
def log_decorator(f):
    """Add logging for method of the model."""
    patch_dict = {"function": f.__name__, "line": inspect.getsourcelines(f)[1], "name": inspect.getmodule(f).__name__}

    @functools.wraps(f)
    def wrapper(self, *args, **kwargs):
        tslogger.log(f"Calling method {f.__name__} of {self.__class__.__name__}", **patch_dict)
        result = f(self, *args, **kwargs)
        return result

    return wrapper
