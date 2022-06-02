import math
from typing import Sequence
from typing import Tuple

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


def prepare_axes(
    num_plots: int, columns_num: int, figsize: Tuple[int, int], set_grid: bool = True
) -> Tuple[matplotlib.figure.Figure, Sequence[matplotlib.axes.Axes]]:
    """Prepare axes according to segments, figure size and number of columns."""
    columns_num = min(columns_num, num_plots)
    rows_num = math.ceil(num_plots / columns_num)

    figsize = (figsize[0] * columns_num, figsize[1] * rows_num)
    fig, ax = plt.subplots(rows_num, columns_num, figsize=figsize, constrained_layout=True)
    ax = np.array([ax]).ravel()

    if set_grid:
        for cur_ax in ax:
            cur_ax.grid()

    return fig, ax
