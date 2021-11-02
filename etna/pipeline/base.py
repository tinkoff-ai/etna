import warnings


def check_support_confidence_interval(is_supported: bool = False, confidence_interval_option: bool = False):
    """Check if pipeline supports confidence intervals, if not, warns a user.

    Parameters
    ----------
    is_supported:
        is model supports intervals
    confidence_interval_option:
        is forecast method called with `confidence_interval=True`
    """
    if not is_supported and confidence_interval_option:
        warnings.warn("This class doesn't support confidence intervals and they won't be build")
