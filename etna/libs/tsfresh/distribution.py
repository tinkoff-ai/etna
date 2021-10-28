def initialize_warnings_in_workers(show_warnings):
    """
    Small helper function to initialize warnings module in multiprocessing workers.

    On Windows, Python spawns fresh processes which do not inherit from warnings
    state, so warnings must be enabled/disabled before running computations.

    :param show_warnings: whether to show warnings or not.
    :type show_warnings: bool
    """
    warnings.catch_warnings()
    if not show_warnings:
        warnings.simplefilter("ignore")
    else:
        warnings.simplefilter("default")