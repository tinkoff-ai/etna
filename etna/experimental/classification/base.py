import pickle


class PickleSerializable:
    """Implementation of serialization using pickle."""

    def dump(self, path: str, *args, **kwargs):
        """Save the object."""
        with open(path, "wb") as file:
            pickle.dump(self, file, *args, **kwargs)

    @staticmethod
    def load(path: str, *args, **kwargs):
        """Load the object."""
        with open(path, "rb") as file:
            clf = pickle.load(file, *args, **kwargs)
        return clf
