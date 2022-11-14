import pathlib
from abc import ABC
from abc import abstractmethod


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
    def load(cls, source: pathlib.Path) -> "AbstractSaveable":
        """Load an object.

        Parameters
        ----------
        source:
            Path to load object from.
        """
        pass
