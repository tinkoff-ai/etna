import configparser
import os
import warnings
from importlib.util import find_spec
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple


def _module_available(module_path: str) -> bool:
    """Check if a path is available in your environment.
    >>> _module_available('os')
    True
    >>> _module_available('bla.bla')
    False
    """
    try:
        return find_spec(module_path) is not None
    except AttributeError:
        # Python 3.6
        return False
    except ModuleNotFoundError:
        # Python 3.7+
        return False


def _is_torch_available():
    true_case = (
        _module_available("pytorch_forecasting") & _module_available("pytorch_lightning") & _module_available("torch")
    )
    if true_case:
        return True
    else:
        warnings.warn("etna[torch] is not available, to install it, run `pip install etna[torch]`")
        return False


def _is_wandb_available():
    if _module_available("wandb"):
        return True
    else:
        warnings.warn("wandb is not available, to install it, run `pip install etna[wandb]`")
        return False


def _is_prophet_available():
    if _module_available("prophet"):
        return True
    else:
        warnings.warn("etna[prophet] is not available, to install it, run `pip install etna[prophet]`")
        return False


def _is_tsfresh_available():
    if _module_available("tsfresh"):
        return True
    else:
        warnings.warn(
            "`tsfresh` is not available, to install it, run `pip install tsfresh==0.19.0 && pip install protobuf==3.20.1`"
        )
        return False


def _get_optional_value(is_required: Optional[bool], is_available_fn: Callable, assert_msg: str) -> bool:
    if is_required is None:
        return is_available_fn()
    elif is_required:
        if not is_available_fn():
            raise ImportError(assert_msg)
        return True
    else:
        return False


class Settings:
    """etna settings."""

    def __init__(  # noqa: D107
        self,
        torch_required: Optional[bool] = None,
        prophet_required: Optional[bool] = None,
        wandb_required: Optional[bool] = None,
        tsfresh_required: Optional[bool] = None,
    ):
        # True – use the package
        # None – use the package if available
        # False - block the package
        self.torch_required: bool = _get_optional_value(
            torch_required,
            _is_torch_available,
            "etna[torch] is not available, to install it, run `pip install etna[torch]`.",
        )
        self.wandb_required: bool = _get_optional_value(
            wandb_required, _is_wandb_available, "wandb is not available, to install it, " "run `pip install wandb`."
        )
        self.prophet_required: bool = _get_optional_value(
            prophet_required,
            _is_prophet_available,
            "etna[prophet] is not available, to install it, run `pip install etna[prophet]`.",
        )
        self.tsfresh_required: bool = _get_optional_value(
            tsfresh_required,
            _is_tsfresh_available,
            "`tsfresh` is not available, to install it, run `pip install tsfresh==0.19.0 && pip install protobuf==3.20.1`",
        )

    @staticmethod
    def parse() -> "Settings":
        """Parse and return the settings.

        Returns
        -------
        Settings:
            Dictionary of the parsed and merged Settings.
        """
        kwargs = MergedConfigParser(ConfigFileFinder("etna")).parse()
        return Settings(**kwargs)

    def type_hint(self, key: str):
        """Return type hint for the specified ``key``.

        Parameters
        ----------
        key:
            key of interest

        Returns
        -------
            type hint for the specified key
        """
        # return get_type_hints(self).get(key, None)
        return type(getattr(self, key, None))


DEFAULT_SETTINGS = Settings()


class ConfigFileFinder:
    """Encapsulate the logic for finding and reading config files.

    Adapted from:

    - https://github.com/catalyst-team/catalyst (Apache-2.0 License)
    """

    def __init__(self, program_name: str) -> None:
        """Initialize object to find config files.

        Parameters
        ----------
        program_name:
            Name of the current program (e.g., catalyst).
        """
        # user configuration file
        self.program_name = program_name
        self.user_config_file = self._user_config_file(program_name)

        # list of filenames to find in the local/project directory
        self.project_filenames = (f".{program_name}",)

        self.local_directory = os.path.abspath(os.curdir)

    @staticmethod
    def _user_config_file(program_name: str) -> str:
        if os.name == "nt":  # if running on Windows
            home_dir = os.path.expanduser("~")
            config_file_basename = f".{program_name}"
        else:
            home_dir = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
            config_file_basename = program_name

        return os.path.join(home_dir, config_file_basename)

    @staticmethod
    def _read_config(*files: str) -> Tuple[configparser.RawConfigParser, List[str]]:
        config = configparser.RawConfigParser()

        found_files: List[str] = []
        for filename in files:
            try:
                found_files.extend(config.read(filename))
            except UnicodeDecodeError:
                print(f"There was an error decoding a config file." f" The file with a problem was {filename}.")
            except configparser.ParsingError:
                print(f"There was an error trying to parse a config file." f" The file with a problem was {filename}.")

        return config, found_files

    def generate_possible_local_files(self):
        """Find and generate all local config files.

        Yields
        ------
        str:
            Path to config file.
        """
        parent = tail = os.getcwd()
        found_config_files = False
        while tail and not found_config_files:
            for project_filename in self.project_filenames:
                filename = os.path.abspath(os.path.join(parent, project_filename))
                if os.path.exists(filename):
                    yield filename
                    found_config_files = True
                    self.local_directory = parent
            (parent, tail) = os.path.split(parent)

    def local_config_files(self) -> List[str]:  # noqa: D202
        """
        Find all local config files which actually exist.

        Returns
        -------
        List[str]:
            List of files that exist that are
            local project config  files with extra config files
            appended to that list (which also exist).
        """
        return list(self.generate_possible_local_files())

    def local_configs(self):
        """Parse all local config files into one config object."""
        config, found_files = self._read_config(*self.local_config_files())
        if found_files:
            print(f"Found local configuration files: {found_files}")
        return config

    def user_config(self):
        """Parse the user config file into a config object."""
        config, found_files = self._read_config(self.user_config_file)
        if found_files:
            print(f"Found user configuration files: {found_files}")
        return config


class MergedConfigParser:
    """Encapsulate merging different types of configuration files.

    This parses out the options registered that were specified in the
    configuration files, handles extra configuration files, and returns
    dictionaries with the parsed values.

    Adapted from:

    - https://github.com/catalyst-team/catalyst (Apache-2.0 License)
    """

    #: Set of actions that should use the
    #: :meth:`~configparser.RawConfigParser.getbool` method.
    GETBOOL_ACTIONS = {"store_true", "store_false"}

    def __init__(self, config_finder: ConfigFileFinder):
        """Initialize the MergedConfigParser instance.

        Parameters
        ----------
        config_finder:
            Initialized ConfigFileFinder.
        """
        self.program_name = config_finder.program_name
        self.config_finder = config_finder

    def _normalize_value(self, option, value):
        final_value = option.normalize(value, self.config_finder.local_directory)
        print(f"{value} has been normalized to {final_value}" f" for option '{option.config_name}'")
        return final_value

    def _parse_config(self, config_parser):
        type2method = {bool: config_parser.getboolean, int: config_parser.getint}

        config_dict: Dict[str, Any] = {}
        if config_parser.has_section(self.program_name):
            for option_name in config_parser.options(self.program_name):
                type_ = DEFAULT_SETTINGS.type_hint(option_name)
                method = type2method.get(type_, config_parser.get)
                config_dict[option_name] = method(self.program_name, option_name)

        return config_dict

    def parse(self) -> dict:
        """Parse and return the local and user config files.

        First this copies over the parsed local configuration and then
        iterates over the options in the user configuration and sets them if
        they were not set by the local configuration file.

        Returns
        -------
        dict:
            Dictionary of the parsed and merged configuration options.
        """
        user_config = self._parse_config(self.config_finder.user_config())
        config = self._parse_config(self.config_finder.local_configs())

        for option, value in user_config.items():
            config.setdefault(option, value)

        return config


SETTINGS = Settings.parse()

__all__ = ["SETTINGS", "Settings", "ConfigFileFinder", "MergedConfigParser"]
