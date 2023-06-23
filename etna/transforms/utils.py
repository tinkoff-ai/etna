import reprlib
from typing import Iterable
from typing import Optional

from etna.datasets.utils import inverse_transform_target_components  # noqa: F401
from etna.datasets.utils import match_target_quantiles  # noqa: F401


def check_new_segments(transform_segments: Iterable[str], fit_segments: Optional[Iterable[str]]):
    """Check if there are any new segments that weren't present during training."""
    if fit_segments is None:
        raise ValueError("Transform is not fitted!")

    new_segments = set(transform_segments) - set(fit_segments)
    if len(new_segments) > 0:
        raise NotImplementedError(
            f"This transform can't process segments that weren't present on train data: {reprlib.repr(new_segments)}"
        )
