import math
from typing import List
from typing import Sequence
from typing import Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np


def prepare_axes(segments: List[str], columns_num: int, figsize: Tuple[int, int]) -> Sequence[matplotlib.axes.Axes]:
    """Prepare axes according to segments, figure size and number of columns."""
    segments_number = len(segments)
    columns_num = min(columns_num, len(segments))
    rows_num = math.ceil(segments_number / columns_num)

    figsize = (figsize[0] * columns_num, figsize[1] * rows_num)
    _, ax = plt.subplots(rows_num, columns_num, figsize=figsize, constrained_layout=True)
    ax = np.array([ax]).ravel()
    return ax
