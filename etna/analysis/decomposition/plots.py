import math
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ruptures.base import BaseCost
from ruptures.base import BaseEstimator
from ruptures.exceptions import BadSegmentationParameters
from statsmodels.tsa.seasonal import STL
from typing_extensions import Literal

from etna.analysis.decomposition.utils import _get_labels_names
from etna.analysis.decomposition.utils import _prepare_seasonal_plot_df
from etna.analysis.decomposition.utils import _seasonal_split
from etna.analysis.utils import _get_borders_ts
from etna.analysis.utils import _prepare_axes

if TYPE_CHECKING:
    from etna.datasets import TSDataset
    from etna.transforms.decomposition import ChangePointsTrendTransform
    from etna.transforms.decomposition import LinearTrendTransform
    from etna.transforms.decomposition import STLTransform
    from etna.transforms.decomposition import TheilSenTrendTransform


TrendTransformType = Union[
    "ChangePointsTrendTransform", "LinearTrendTransform", "TheilSenTrendTransform", "STLTransform"
]


def plot_trend(
    ts: "TSDataset",
    trend_transform: Union[TrendTransformType, List[TrendTransformType]],
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot series and trend from trend transform for this series.

    If only unique transform classes are used then show their short names (without parameters).
    Otherwise show their full repr as label

    Parameters
    ----------
    ts:
        dataframe of timeseries that was used for trend plot
    trend_transform:
        trend transform or list of trend transforms to apply
    segments:
        segments to use
    columns_num:
        number of columns in subplots
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if segments is None:
        segments = ts.segments

    _, ax = _prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)
    df = ts.df

    if not isinstance(trend_transform, list):
        trend_transform = [trend_transform]

    df_detrend = [transform.fit_transform(deepcopy(ts)).to_pandas() for transform in trend_transform]
    labels, linear_coeffs = _get_labels_names(trend_transform, segments)

    for i, segment in enumerate(segments):
        ax[i].plot(df[segment]["target"], label="Initial series")
        for label, df_now in zip(labels, df_detrend):
            ax[i].plot(df[segment, "target"] - df_now[segment, "target"], label=label + linear_coeffs[segment], lw=3)
        ax[i].set_title(segment)
        ax[i].tick_params("x", rotation=45)
        ax[i].legend()


def plot_time_series_with_change_points(
    ts: "TSDataset",
    change_points: Dict[str, List[pd.Timestamp]],
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """Plot segments with their trend change points.

    Parameters
    ----------
    ts:
        TSDataset with timeseries
    change_points:
        dictionary with trend change points for each segment,
        can be obtained from :py:func:`~etna.analysis.decomposition.search.find_change_points`
    segments:
        segments to use
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
        change_points_segment = change_points[segment]

        # plot each part of segment separately
        timestamp = segment_df.index.values
        target = segment_df["target"].values
        change_points_segment = [
            i for i in change_points_segment if pd.Timestamp(timestamp[0]) < i < pd.Timestamp(timestamp[-1])
        ]
        all_change_points_segment = [pd.Timestamp(timestamp[0])] + change_points_segment + [pd.Timestamp(timestamp[-1])]
        for idx in range(len(all_change_points_segment) - 1):
            start_time = all_change_points_segment[idx]
            end_time = all_change_points_segment[idx + 1]
            selected_indices = (timestamp >= start_time) & (timestamp <= end_time)
            cur_timestamp = timestamp[selected_indices]
            cur_target = target[selected_indices]
            ax[i].plot(cur_timestamp, cur_target)

        # plot each trend change point
        for change_point in change_points_segment:
            ax[i].axvline(change_point, linestyle="dashed", c="grey")

        ax[i].set_title(segment)
        ax[i].tick_params("x", rotation=45)


def plot_change_points_interactive(
    ts,
    change_point_model: BaseEstimator,
    model: BaseCost,
    params_bounds: Dict[str, Tuple[Union[int, float], Union[int, float], Union[int, float]]],
    model_params: List[str],
    predict_params: List[str],
    in_column: str = "target",
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """Plot a time series with indicated change points.

    Change points are obtained using the specified method. The method parameters values
    can be changed using the corresponding sliders.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    change_point_model:
        model to get trend change points
    model:
        binseg segment model, ["l1", "l2", "rbf",...]. Not used if 'custom_cost' is not None
    params_bounds:
        Parameters ranges of the change points detection. Bounds for the parameter are (min,max,step)
    model_params:
        List of iterable parameters for initialize the model
    predict_params:
        List of iterable parameters for predict method
    in_column:
        column to plot
    segments:
        segments to use
    columns_num:
        number of subplots columns
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
    >>> from etna.analysis import plot_change_points_interactive
    >>> from ruptures.detection import Binseg
    >>> classic_df = generate_ar_df(periods=1000, start_time="2021-08-01", n_segments=2)
    >>> df = TSDataset.to_dataset(classic_df)
    >>> ts = TSDataset(df, "D")
    >>> params_bounds = {"n_bkps": [0, 5, 1], "min_size":[1,10,3]}
    >>> plot_change_points_interactive(ts=ts, change_point_model=Binseg, model="l2", params_bounds=params_bounds, model_params=["min_size"], predict_params=["n_bkps"], figsize=(20, 10)) # doctest: +SKIP
    """
    from ipywidgets import FloatSlider
    from ipywidgets import IntSlider
    from ipywidgets import interact

    if segments is None:
        segments = sorted(ts.segments)

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
        _, ax = _prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)

        key = "_".join([str(val) for val in kwargs.values()])

        is_fitted = False

        if key not in cache:
            m_params = {x: kwargs[x] for x in model_params}
            p_params = {x: kwargs[x] for x in predict_params}
            cache[key] = {}
        else:
            is_fitted = True

        for i, segment in enumerate(segments):
            ax[i].cla()
            segment_df = ts[start:end, segment, :][segment]
            timestamp = segment_df.index.values
            target = segment_df[in_column].values

            if not is_fitted:
                try:
                    algo = change_point_model(model=model, **m_params).fit(signal=target)
                    bkps = algo.predict(**p_params)
                    cache[key][segment] = bkps
                    cache[key][segment].insert(0, 1)
                except BadSegmentationParameters:
                    cache[key][segment] = None

            segment_bkps = cache[key][segment]

            if segment_bkps is not None:
                for idx in range(len(segment_bkps[:-1])):
                    bkp = segment_bkps[idx] - 1
                    start_time = timestamp[bkp]
                    end_time = timestamp[segment_bkps[idx + 1] - 1]
                    selected_indices = (timestamp >= start_time) & (timestamp <= end_time)
                    cur_timestamp = timestamp[selected_indices]
                    cur_target = target[selected_indices]
                    ax[i].plot(cur_timestamp, cur_target)
                    if bkp != 0:
                        ax[i].axvline(timestamp[bkp], linestyle="dashed", c="grey")

            else:
                box = {"facecolor": "grey", "edgecolor": "red", "boxstyle": "round"}
                ax[i].text(
                    0.5, 0.4, "Parameters\nError", bbox=box, horizontalalignment="center", color="white", fontsize=50
                )
            ax[i].set_title(segment)
            ax[i].tick_params("x", rotation=45)
        plt.show()

    interact(update, **sliders)


def stl_plot(
    ts: "TSDataset",
    period: int,
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 10),
    plot_kwargs: Optional[Dict[str, Any]] = None,
    stl_kwargs: Optional[Dict[str, Any]] = None,
):
    """Plot STL decomposition for segments.

    Parameters
    ----------
    ts:
        dataset with timeseries data
    period:
        length of seasonality
    segments:
        segments to plot
    columns_num:
        number of columns in subplots
    figsize:
        size of the figure per subplot with one segment in inches
    plot_kwargs:
        dictionary with parameters for plotting, :py:meth:`matplotlib.axes.Axes.plot` is used
    stl_kwargs:
        dictionary with parameters for STL decomposition, :py:class:`statsmodels.tsa.seasonal.STL` is used
    """
    if plot_kwargs is None:
        plot_kwargs = {}
    if stl_kwargs is None:
        stl_kwargs = {}
    if segments is None:
        segments = sorted(ts.segments)

    in_column = "target"

    segments_number = len(segments)
    columns_num = min(columns_num, len(segments))
    rows_num = math.ceil(segments_number / columns_num)

    figsize = (figsize[0] * columns_num, figsize[1] * rows_num)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    subfigs = fig.subfigures(rows_num, columns_num, squeeze=False)

    df = ts.to_pandas()
    for i, segment in enumerate(segments):
        segment_df = df.loc[:, pd.IndexSlice[segment, :]][segment]
        segment_df = segment_df[segment_df.first_valid_index() : segment_df.last_valid_index()]
        decompose_result = STL(endog=segment_df[in_column], period=period, **stl_kwargs).fit()

        # start plotting
        subfigs.flat[i].suptitle(segment)
        axs = subfigs.flat[i].subplots(4, 1, sharex=True)

        # plot observed
        axs.flat[0].plot(segment_df.index, decompose_result.observed, **plot_kwargs)
        axs.flat[0].set_ylabel("Observed")
        axs.flat[0].grid()

        # plot trend
        axs.flat[1].plot(segment_df.index, decompose_result.trend, **plot_kwargs)
        axs.flat[1].set_ylabel("Trend")
        axs.flat[1].grid()

        # plot seasonal
        axs.flat[2].plot(segment_df.index, decompose_result.seasonal, **plot_kwargs)
        axs.flat[2].set_ylabel("Seasonal")
        axs.flat[2].grid()

        # plot residuals
        axs.flat[3].plot(segment_df.index, decompose_result.resid, **plot_kwargs)
        axs.flat[3].set_ylabel("Residual")
        axs.flat[3].tick_params("x", rotation=45)
        axs.flat[3].grid()


def seasonal_plot(
    ts: "TSDataset",
    freq: Optional[str] = None,
    cycle: Union[
        Literal["hour"], Literal["day"], Literal["week"], Literal["month"], Literal["quarter"], Literal["year"], int
    ] = "year",
    alignment: Union[Literal["first"], Literal["last"]] = "last",
    aggregation: Union[Literal["sum"], Literal["mean"]] = "sum",
    in_column: str = "target",
    plot_params: Optional[Dict[str, Any]] = None,
    cmap: str = "plasma",
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
):
    """Plot each season on one canvas for each segment.

    Parameters
    ----------
    ts:
        dataset with timeseries data
    freq:
        frequency to analyze seasons:

        * if isn't set, the frequency of ``ts`` will be used;

        * if set, resampling will be made using ``aggregation`` parameter.
          If given frequency is too low, then the frequency of ``ts`` will be used.

    cycle:
        period of seasonality to capture (see :class:`~etna.analysis.decomposition.utils.SeasonalPlotCycle`)
    alignment:
        how to align dataframe in case of integer cycle (see :py:class:`~etna.analysis.decomposition.utils.SeasonalPlotAlignment`)
    aggregation:
        how to aggregate values during resampling (see :py:class:`~etna.analysis.decomposition.utils.SeasonalPlotAggregation`)
    in_column:
        column to use
    cmap:
        name of colormap for plotting different cycles
        (see `Choosing Colormaps in Matplotlib <https://matplotlib.org/3.5.1/tutorials/colors/colormaps.html>`_)
    plot_params:
        dictionary with parameters for plotting, :py:meth:`matplotlib.axes.Axes.plot` is used
    segments:
        segments to use
    columns_num:
        number of columns in subplots
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if plot_params is None:
        plot_params = {}
    if freq is None:
        freq = ts.freq
    if segments is None:
        segments = sorted(ts.segments)

    df = _prepare_seasonal_plot_df(
        ts=ts,
        freq=freq,
        cycle=cycle,
        alignment=alignment,
        aggregation=aggregation,
        in_column=in_column,
        segments=segments,
    )
    seasonal_df = _seasonal_split(timestamp=df.index.to_series(), freq=freq, cycle=cycle)

    colors = plt.get_cmap(cmap)
    _, ax = _prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)
    for i, segment in enumerate(segments):
        segment_df = df.loc[:, pd.IndexSlice[segment, "target"]]
        cycle_names = seasonal_df["cycle_name"].unique()
        for j, cycle_name in enumerate(cycle_names):
            color = colors(j / len(cycle_names))
            cycle_df = seasonal_df[seasonal_df["cycle_name"] == cycle_name]
            segment_cycle_df = segment_df.loc[cycle_df["timestamp"]]
            ax[i].plot(
                cycle_df["in_cycle_num"],
                segment_cycle_df[cycle_df["timestamp"]],
                color=color,
                label=cycle_name,
                **plot_params,
            )

        # draw ticks if they are not digits
        if not np.all(seasonal_df["in_cycle_name"].str.isnumeric()):
            ticks_dict = dict(zip(seasonal_df["in_cycle_num"], seasonal_df["in_cycle_name"]))
            ticks = np.array(list(ticks_dict.keys()))
            ticks_labels = np.array(list(ticks_dict.values()))
            idx_sort = np.argsort(ticks)
            ax[i].set_xticks(ticks=ticks[idx_sort], labels=ticks_labels[idx_sort])
        ax[i].set_xlabel(freq)
        ax[i].set_title(segment)
        ax[i].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=6)
