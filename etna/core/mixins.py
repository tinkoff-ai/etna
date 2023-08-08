import inspect
import json
import pathlib
import sys
import warnings
import zipfile
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import cast

import dill
import hydra_slayer
from sklearn.base import BaseEstimator
from typing_extensions import Self

from etna.core.saving import AbstractSaveable

TMixin = TypeVar("TMixin", bound="BaseMixin")


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
    def _get_target_from_function(value: Callable):
        return str(value.__module__) + "." + str(value.__qualname__)

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
        # this if is for objects imported from etna.libs.pytorch_lightning.callbacks
        elif hasattr(value, "_init_params"):
            return {"_target_": BaseMixin._get_target_from_class(value), **value._init_params}
        elif isinstance(value, (str, float, int)):
            return value
        elif isinstance(value, list):
            return [BaseMixin._parse_value(elem) for elem in value]
        elif isinstance(value, tuple):
            return tuple([BaseMixin._parse_value(elem) for elem in value])
        elif isinstance(value, dict):
            return {key: BaseMixin._parse_value(item) for key, item in value.items()}
        elif inspect.isfunction(value):
            return {"_target_": BaseMixin._get_target_from_function(value)}
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

    @classmethod
    def _update_nested_structure(cls, structure: Any, keys: Sequence[str], value: Any) -> Any:
        """Update nested structure by sequence of keys with value.

        Method applies sequence of keys to structure and returns the structure with changed value.
        Key can only be applied to ``dict``, ``list`` or ``tuple``.
        For ``list`` and ``tuple`` function ``int`` is used to make index from the key.

        Raises
        ------
        ValueError:
            Unsupported type of structure to update
        """
        if len(keys) == 0:
            return value

        current_key = keys[0]
        new_structure: Any
        if isinstance(structure, dict):
            structure_to_update = structure.get(current_key, {})
            current_value = cls._update_nested_structure(structure_to_update, keys[1:], value)
            new_structure = structure
            new_structure[current_key] = current_value
        elif isinstance(structure, list):
            idx = int(current_key)
            structure_to_update = structure[idx]
            current_value = cls._update_nested_structure(structure_to_update, keys[1:], value)
            new_structure = structure
            new_structure[idx] = current_value
        elif isinstance(structure, tuple):
            idx = int(current_key)
            structure_to_update = structure[idx]
            current_value = cls._update_nested_structure(structure_to_update, keys[1:], value)
            new_temp_structure = list(structure)
            new_temp_structure[idx] = current_value
            new_structure = tuple(new_temp_structure)
        else:
            raise ValueError(
                f"Structure to update is {structure} with type {type(structure)}, allowed types are dict, list, tuple"
            )

        return new_structure

    def set_params(self: TMixin, **params: dict) -> TMixin:
        """Return new object instance with modified parameters.

        Method also allows to change parameters of nested objects within the current object.
        For example, it is possible to change parameters of a ``model`` in a :class:`~etna.pipeline.Pipeline`.

        Nested parameters are expected to be in a ``<component_1>.<...>.<parameter>`` form,
        where components are separated by a dot.

        Parameters
        ----------
        **params:
            Estimator parameters

        Returns
        -------
        :
            New instance with changed parameters

        Examples
        --------
        >>> from etna.pipeline import Pipeline
        >>> from etna.models import NaiveModel
        >>> from etna.transforms import AddConstTransform
        >>> model = model=NaiveModel(lag=1)
        >>> transforms = [AddConstTransform(in_column="target", value=1)]
        >>> pipeline = Pipeline(model, transforms=transforms, horizon=3)
        >>> pipeline.set_params(**{"model.lag": 3, "transforms.0.value": 2})
        Pipeline(model = NaiveModel(lag = 3, ), transforms = [AddConstTransform(in_column = 'target', value = 2, inplace = True, out_column = None, )], horizon = 3, )
        """
        params_dict = self.to_dict()

        new_params_dict = params_dict
        for current_key, value in params.items():
            keys = current_key.split(".")
            new_params_dict = self._update_nested_structure(new_params_dict, keys, value)

        estimator_out = hydra_slayer.get_from_params(**new_params_dict)
        return estimator_out


class StringEnumWithRepr(str, Enum):
    """Base class for str enums, that has alternative __repr__ method."""

    def __repr__(self):
        """Get string representation for enum string so that enum can be created from it."""
        return self.value.__repr__()


def get_etna_version() -> Tuple[int, int, int]:
    """Get current version of etna library."""
    python_version = sys.version_info
    if python_version[0] == 3 and python_version[1] >= 8:
        from importlib.metadata import version

        str_version = version("etna")
        result = tuple([int(x) for x in str_version.split(".")])
        result = cast(Tuple[int, int, int], result)
        return result
    else:
        import pkg_resources

        str_version = pkg_resources.get_distribution("etna").version
        result = tuple([int(x) for x in str_version.split(".")])
        result = cast(Tuple[int, int, int], result)
        return result


class SaveMixin(AbstractSaveable):
    """Basic implementation of ``AbstractSaveable`` abstract class.

    It saves object to the zip archive with 2 files:

    * metadata.json: contains library version and class name.

    * object.pkl: pickled object.
    """

    def _save_metadata(self, archive: zipfile.ZipFile):
        full_class_name = f"{inspect.getmodule(self).__name__}.{self.__class__.__name__}"  # type: ignore
        metadata = {
            "etna_version": get_etna_version(),
            "class": full_class_name,
        }
        metadata_str = json.dumps(metadata, indent=2, sort_keys=True)
        metadata_bytes = metadata_str.encode("utf-8")
        with archive.open("metadata.json", "w") as output_file:
            output_file.write(metadata_bytes)

    def _save_state(self, archive: zipfile.ZipFile):
        with archive.open("object.pkl", "w", force_zip64=True) as output_file:
            dill.dump(self, output_file)

    def save(self, path: pathlib.Path):
        """Save the object.

        Parameters
        ----------
        path:
            Path to save object to.
        """
        with zipfile.ZipFile(path, "w") as archive:
            self._save_metadata(archive)
            self._save_state(archive)

    @classmethod
    def _load_metadata(cls, archive: zipfile.ZipFile) -> Dict[str, Any]:
        with archive.open("metadata.json", "r") as input_file:
            metadata_bytes = input_file.read()
        metadata_str = metadata_bytes.decode("utf-8")
        metadata = json.loads(metadata_str)
        return metadata

    @classmethod
    def _validate_metadata(cls, metadata: Dict[str, Any]):
        current_etna_version = get_etna_version()
        saved_etna_version = tuple(metadata["etna_version"])

        # if major version is different give a warning
        if current_etna_version[0] != saved_etna_version[0] or current_etna_version[:2] < saved_etna_version[:2]:
            current_etna_version_str = ".".join([str(x) for x in current_etna_version])
            saved_etna_version_str = ".".join([str(x) for x in saved_etna_version])
            warnings.warn(
                f"The object was saved under etna version {saved_etna_version_str} "
                f"but running version is {current_etna_version_str}, this can cause problems with compatibility!"
            )

    @classmethod
    def _load_state(cls, archive: zipfile.ZipFile) -> Self:
        with archive.open("object.pkl", "r") as input_file:
            return dill.load(input_file)

    @classmethod
    def load(cls, path: pathlib.Path) -> Self:
        """Load an object.

        Warning
        -------
        This method uses :py:mod:`dill` module which is not secure.
        It is possible to construct malicious data which will execute arbitrary code during loading.
        Never load data that could have come from an untrusted source, or that could have been tampered with.

        Parameters
        ----------
        path:
            Path to load object from.

        Returns
        -------
        :
            Loaded object.
        """
        with zipfile.ZipFile(path, "r") as archive:
            metadata = cls._load_metadata(archive)
            cls._validate_metadata(metadata)
            obj = cls._load_state(archive)
        return obj
