import pathlib
from abc import ABC
from abc import abstractmethod
from typing import Any


class AbstractSaveable(ABC):
    """Abstract class with methods for saving, loading objects."""

    @abstractmethod
    def save(self, path: pathlib.Path):
        """Save the object.

        Parameters
        ----------
        path:
            Path to save object to.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: pathlib.Path) -> Any:
        """Load an object.

        Parameters
        ----------
        path:
            Path to load object from.
        """
        pass
