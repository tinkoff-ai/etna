import pickle


class PickleSerializable:
    """Implementation of serialization using pickle."""

    def dump(self, path: str, *args, **kwargs):
        """Save the object."""
        with open(path, "wb") as file:
            pickle.dump(self, file, *args, **kwargs)

    @staticmethod
    def load(path: str, *args, **kwargs):
        """Load the object.

        Warning
        -------
        This method uses :py:mod:`dill` module which is not secure.
        It is possible to construct malicious data which will execute arbitrary code during loading.
        Never load data that could have come from an untrusted source, or that could have been tampered with.
        """
        with open(path, "rb") as file:
            clf = pickle.load(file, *args, **kwargs)
        return clf
