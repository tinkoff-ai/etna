from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

from etna.analysis.utils import _get_borders_ts
from etna.analysis.utils import _prepare_axes

if TYPE_CHECKING:
    from etna.datasets import TSDataset


def plot_anomalies(
    ts: "TSDataset",
    anomaly_dict: Dict[str, List[pd.Timestamp]],
    in_column: str = "target",
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """Plot a time series with indicated anomalies.

    Parameters
    ----------
    ts:
        TSDataset of timeseries that was used for detect anomalies
    anomaly_dict:
        dictionary derived from anomaly detection function,
        e.g. :py:func:`~etna.analysis.outliers.density_outliers.get_anomalies_density`
    in_column:
        column to plot
    segments:
        segments to plot
    columns_num:
        number of subplots columns
    figsize:
        size of the figure per subplot with one segment in inches
    start:
        start timestamp for plot
    end:
        end timestamp for plot
    """
    start, end = _get_borders_ts(ts, start, end)

    if segments is None:
        segments = sorted(ts.segments)

    _, ax = _prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)

    for i, segment in enumerate(segments):
        segment_df = ts[start:end, segment, :][segment]  # type: ignore
        anomaly = anomaly_dict[segment]

        ax[i].set_title(segment)
        ax[i].plot(segment_df.index.values, segment_df[in_column].values)

        anomaly = [i for i in sorted(anomaly) if i in segment_df.index]  # type: ignore
        ax[i].scatter(anomaly, segment_df[segment_df.index.isin(anomaly)][in_column].values, c="r")

        ax[i].tick_params("x", rotation=45)


def plot_anomalies_interactive(
    ts: "TSDataset",
    segment: str,
    method: Callable[..., Dict[str, List[pd.Timestamp]]],
    params_bounds: Dict[str, Tuple[Union[int, float], Union[int, float], Union[int, float]]],
    in_column: str = "target",
    figsize: Tuple[int, int] = (20, 10),
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """Plot a time series with indicated anomalies.

    Anomalies are obtained using the specified method. The method parameters values
    can be changed using the corresponding sliders.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    segment:
        Segment to plot
    method:
        Method for outliers detection, e.g. :py:func:`~etna.analysis.outliers.density_outliers.get_anomalies_density`
    params_bounds:
        Parameters ranges of the outliers detection method. Bounds for the parameter are (min,max,step)
    in_column:
        column to plot
    figsize:
        size of the figure in inches
    start:
        start timestamp for plot
    end:
        end timestamp for plot

    Notes
    -----
    Jupyter notebook might display the results incorrectly,
    in this case try to use ``!jupyter nbextension enable --py widgetsnbextension``.

    Examples
    --------
    >>> from etna.datasets import TSDataset
    >>> from etna.datasets import generate_ar_df
    >>> from etna.analysis import plot_anomalies_interactive, get_anomalies_density
    >>> classic_df = generate_ar_df(periods=1000, start_time="2021-08-01", n_segments=2)
    >>> df = TSDataset.to_dataset(classic_df)
    >>> ts = TSDataset(df, "D")
    >>> params_bounds = {"window_size": (5, 20, 1), "distance_coef": (0.1, 3, 0.25)}
    >>> method = get_anomalies_density
    >>> plot_anomalies_interactive(ts=ts, segment="segment_1", method=method, params_bounds=params_bounds, figsize=(20, 10)) # doctest: +SKIP
    """
    from ipywidgets import FloatSlider
    from ipywidgets import IntSlider
    from ipywidgets import interact

    from etna.datasets import TSDataset

    start, end = _get_borders_ts(ts, start, end)

    df = ts[start:end, segment, in_column]  # type: ignore
    ts = TSDataset(ts[:, segment, :], ts.freq)
    x, y = df.index.values, df.values
    cache = {}

    sliders = dict()
    style = {"description_width": "initial"}
    for param, bounds in params_bounds.items():
        min_, max_, step = bounds
        if isinstance(min_, float) or isinstance(max_, float) or isinstance(step, float):
            sliders[param] = FloatSlider(min=min_, max=max_, step=step, continuous_update=False, style=style)
        else:
            sliders[param] = IntSlider(min=min_, max=max_, step=step, continuous_update=False, style=style)

    def update(**kwargs):
        key = "_".join([str(val) for val in kwargs.values()])
        if key not in cache:
            anomalies = method(ts, **kwargs)[segment]
            anomalies = [i for i in sorted(anomalies) if i in df.index]
            cache[key] = anomalies
        else:
            anomalies = cache[key]
        plt.figure(figsize=figsize)
        plt.cla()
        plt.plot(x, y)
        plt.scatter(anomalies, y[pd.to_datetime(x).isin(anomalies)], c="r")
        plt.xticks(rotation=45)
        plt.grid()
        plt.show()

    interact(update, **sliders)
