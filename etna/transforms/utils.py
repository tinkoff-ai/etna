import re
import reprlib
from typing import List
from typing import Optional
from typing import Set


def match_target_quantiles(features: Set[str]) -> Set[str]:
    """Find quantiles in dataframe columns."""
    pattern = re.compile("target_\d+\.\d+$")
    return {i for i in list(features) if pattern.match(i) is not None}


def check_new_segments(transform_segments: List[str], fit_segments: Optional[List[str]]):
    """Check if there are any new segments that weren't present during training."""
    if fit_segments is None:
        raise ValueError("Transform is not fitted!")

    new_segments = set(transform_segments) - set(fit_segments)
    if len(new_segments) > 0:
        raise NotImplementedError(
            f"This transform can't process segments that weren't present on train data: {reprlib.repr(new_segments)}"
        )
