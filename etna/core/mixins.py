import inspect
import warnings
from enum import Enum
from typing import Any
from typing import Dict
from typing import List

from sklearn.base import BaseEstimator


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

    @staticmethod
    def _get_target_from_class(value: Any):
        if value is None:
            return None
        return str(value.__module__) + "." + str(value.__class__.__name__)

    @staticmethod
    def _parse_value(value: Any) -> Any:
        if isinstance(value, BaseMixin):
            return value.to_dict()
        elif isinstance(value, BaseEstimator):
            answer = {}
            answer["_target_"] = BaseMixin._get_target_from_class(value)
            model_parameters = value.get_params()
            answer.update(model_parameters)
            return answer
        elif isinstance(value, (str, float, int)):
            return value
        elif isinstance(value, List):
            return [BaseMixin._parse_value(elem) for elem in value]
        elif isinstance(value, tuple):
            return tuple([BaseMixin._parse_value(elem) for elem in value])
        elif isinstance(value, Dict):
            return {key: BaseMixin._parse_value(item) for key, item in value.items()}
        else:
            answer = {}
            answer["_target_"] = BaseMixin._get_target_from_class(value)
            warnings.warn("Some of external objects in input parameters could be not written in dict")
            return answer

    def to_dict(self):
        """Collect all information about etna object in dict."""
        init_args = inspect.signature(self.__init__).parameters
        params = {}
        for arg in init_args.keys():
            value = self.__dict__[arg]
            if value is None:
                continue
            params[arg] = BaseMixin._parse_value(value=value)
        params["_target_"] = BaseMixin._get_target_from_class(self)
        return params


class StringEnumWithRepr(str, Enum):
    """Base class for str enums, that has alternative __repr__ method."""

    def __repr__(self):
        """Get string representation for enum string so that enum can be created from it."""
        return self.value.__repr__()
