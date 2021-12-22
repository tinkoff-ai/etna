import re
from typing import Set


def match_target_quantiles(features: Set[str]) -> Set[str]:
    """Find quantiles in dataframe columns."""
    pattern = re.compile("target_\d+\.\d+$")
    return set(i for i in list(features) if pattern.match(i) is not None)
