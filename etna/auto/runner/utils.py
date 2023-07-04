import dill


def run_dill_encoded(payload: str):
    """Unpickle paylod and call function."""
    fun, args, kwargs = dill.loads(payload)
    return fun(*args, **kwargs)
