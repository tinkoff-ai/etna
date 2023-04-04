from enum import Enum
from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from typing_extensions import Literal

if TYPE_CHECKING:
    from etna.datasets import TSDataset


def _get_labels_names(trend_transform, segments):
    """If only unique transform classes are used then show their short names (without parameters). Otherwise show their full repr as label."""
    from etna.transforms.decomposition import LinearTrendTransform
    from etna.transforms.decomposition import TheilSenTrendTransform

    labels = [transform.__repr__() for transform in trend_transform]
    labels_short = [i[: i.find("(")] for i in labels]
    if len(np.unique(labels_short)) == len(labels_short):
        labels = labels_short
    linear_coeffs = dict(zip(segments, ["" for i in range(len(segments))]))
    if (
        len(trend_transform) == 1
        and isinstance(trend_transform[0], (LinearTrendTransform, TheilSenTrendTransform))
        and trend_transform[0].poly_degree == 1
    ):
        for seg in segments:
            linear_coeffs[seg] = (
                ", k=" + f"{trend_transform[0].segment_transforms[seg]._pipeline.steps[1][1].coef_[0]:g}"
            )
    return labels, linear_coeffs


class SeasonalPlotAlignment(str, Enum):
    """Enum for types of alignment in a seasonal plot.

    Attributes
    ----------
    first:
        make first period full, allow last period to have NaNs in the ending
    last:
        make last period full, allow first period to have NaNs in the beginning
    """

    first = "first"
    last = "last"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} alignments are allowed"
        )


class SeasonalPlotAggregation(str, Enum):
    """Enum for types of aggregation in a seasonal plot."""

    mean = "mean"
    sum = "sum"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} aggregations are allowed"
        )

    @staticmethod
    def _modified_nansum(series):
        """Sum values with ignoring of NaNs.

        * If there some nan: we skip them.

        * If all values equal to nan we return nan.
        """
        if np.all(np.isnan(series)):
            return np.NaN
        else:
            return np.nansum(series)

    def get_function(self):
        """Get aggregation function."""
        if self.value == "mean":
            return np.nanmean
        elif self.value == "sum":
            return self._modified_nansum


class SeasonalPlotCycle(str, Enum):
    """Enum for types of cycles in a seasonal plot."""

    hour = "hour"
    day = "day"
    week = "week"
    month = "month"
    quarter = "quarter"
    year = "year"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} cycles are allowed"
        )


def _get_seasonal_cycle_name(
    timestamp: pd.Series,
    cycle: Union[
        Literal["hour"], Literal["day"], Literal["week"], Literal["month"], Literal["quarter"], Literal["year"], int
    ],
) -> pd.Series:
    """Get unique name for each cycle in a series with timestamps."""
    cycle_functions: Dict[SeasonalPlotCycle, Callable[[pd.Series], pd.Series]] = {
        SeasonalPlotCycle.hour: lambda x: x.dt.strftime("%Y-%m-%d %H"),
        SeasonalPlotCycle.day: lambda x: x.dt.strftime("%Y-%m-%d"),
        SeasonalPlotCycle.week: lambda x: x.dt.strftime("%Y-%W"),
        SeasonalPlotCycle.month: lambda x: x.dt.strftime("%Y-%b"),
        SeasonalPlotCycle.quarter: lambda x: x.apply(lambda x: f"{x.year}-{x.quarter}"),
        SeasonalPlotCycle.year: lambda x: x.dt.strftime("%Y"),
    }

    if isinstance(cycle, int):
        row_numbers = pd.Series(np.arange(len(timestamp)))
        return (row_numbers // cycle + 1).astype(str)
    else:
        return cycle_functions[SeasonalPlotCycle(cycle)](timestamp)


def _get_seasonal_in_cycle_num(
    timestamp: pd.Series,
    cycle_name: pd.Series,
    cycle: Union[
        Literal["hour"], Literal["day"], Literal["week"], Literal["month"], Literal["quarter"], Literal["year"], int
    ],
    freq: str,
) -> pd.Series:
    """Get number for each point within cycle in a series of timestamps."""
    cycle_functions: Dict[Tuple[SeasonalPlotCycle, str], Callable[[pd.Series], pd.Series]] = {
        (SeasonalPlotCycle.hour, "T"): lambda x: x.dt.minute,
        (SeasonalPlotCycle.day, "H"): lambda x: x.dt.hour,
        (SeasonalPlotCycle.week, "D"): lambda x: x.dt.weekday,
        (SeasonalPlotCycle.month, "D"): lambda x: x.dt.day,
        (SeasonalPlotCycle.quarter, "D"): lambda x: (x - pd.PeriodIndex(x, freq="Q").start_time).dt.days,
        (SeasonalPlotCycle.year, "D"): lambda x: x.dt.dayofyear,
        (SeasonalPlotCycle.year, "Q"): lambda x: x.dt.quarter,
        (SeasonalPlotCycle.year, "QS"): lambda x: x.dt.quarter,
        (SeasonalPlotCycle.year, "M"): lambda x: x.dt.month,
        (SeasonalPlotCycle.year, "MS"): lambda x: x.dt.month,
    }

    if isinstance(cycle, int):
        pass
    else:
        key = (SeasonalPlotCycle(cycle), freq)
        if key in cycle_functions:
            return cycle_functions[key](timestamp)

    # in all other cases we can use numbers within each group
    cycle_df = pd.DataFrame({"timestamp": timestamp.tolist(), "cycle_name": cycle_name.tolist()})
    return cycle_df.sort_values("timestamp").groupby("cycle_name").cumcount()


def _get_seasonal_in_cycle_name(
    timestamp: pd.Series,
    in_cycle_num: pd.Series,
    cycle: Union[
        Literal["hour"], Literal["day"], Literal["week"], Literal["month"], Literal["quarter"], Literal["year"], int
    ],
    freq: str,
) -> pd.Series:
    """Get unique name for each point within the cycle in a series of timestamps."""
    if isinstance(cycle, int):
        pass
    elif SeasonalPlotCycle(cycle) == SeasonalPlotCycle.week:
        if freq == "D":
            return timestamp.dt.strftime("%a")
    elif SeasonalPlotCycle(cycle) == SeasonalPlotCycle.year:
        if freq == "M" or freq == "MS":
            return timestamp.dt.strftime("%b")

    # in all other cases we can use numbers from cycle_num
    return in_cycle_num.astype(str)


def _seasonal_split(
    timestamp: pd.Series,
    freq: str,
    cycle: Union[
        Literal["hour"], Literal["day"], Literal["week"], Literal["month"], Literal["quarter"], Literal["year"], int
    ],
) -> pd.DataFrame:
    """Create a seasonal split into cycles of a given timestamp.

    Parameters
    ----------
    timestamp:
        series with timestamps
    freq:
        frequency of dataframe
    cycle:
        period of seasonality to capture (see :py:class:`~etna.analysis.decomposition.utils.SeasonalPlotCycle`)

    Returns
    -------
    result: pd.DataFrame
        dataframe with timestamps and corresponding cycle names and in cycle names
    """
    cycles_df = pd.DataFrame({"timestamp": timestamp.tolist()})
    cycles_df["cycle_name"] = _get_seasonal_cycle_name(timestamp=cycles_df["timestamp"], cycle=cycle)
    cycles_df["in_cycle_num"] = _get_seasonal_in_cycle_num(
        timestamp=cycles_df["timestamp"], cycle_name=cycles_df["cycle_name"], cycle=cycle, freq=freq
    )
    cycles_df["in_cycle_name"] = _get_seasonal_in_cycle_name(
        timestamp=cycles_df["timestamp"], in_cycle_num=cycles_df["in_cycle_num"], cycle=cycle, freq=freq
    )
    return cycles_df


def _resample(df: pd.DataFrame, freq: str, aggregation: Union[Literal["sum"], Literal["mean"]]) -> pd.DataFrame:
    from etna.datasets import TSDataset

    agg_enum = SeasonalPlotAggregation(aggregation)
    df_flat = TSDataset.to_flatten(df)
    df_flat = (
        df_flat.set_index("timestamp")
        .groupby(["segment", pd.Grouper(freq=freq)])
        .agg(agg_enum.get_function())
        .reset_index()
    )
    df = TSDataset.to_dataset(df_flat)
    return df


def _prepare_seasonal_plot_df(
    ts: "TSDataset",
    freq: str,
    cycle: Union[
        Literal["hour"], Literal["day"], Literal["week"], Literal["month"], Literal["quarter"], Literal["year"], int
    ],
    alignment: Union[Literal["first"], Literal["last"]],
    aggregation: Union[Literal["sum"], Literal["mean"]],
    in_column: str,
    segments: List[str],
):
    # for simplicity we will rename our column to target
    df = ts.to_pandas().loc[:, pd.IndexSlice[segments, in_column]]
    df.rename(columns={in_column: "target"}, inplace=True)

    # remove timestamps with only nans, it is possible if in_column != "target"
    df = df[(~df.isna()).sum(axis=1) > 0]

    # make resampling if necessary
    if ts.freq != freq:
        df = _resample(df=df, freq=freq, aggregation=aggregation)

    # process alignment
    if isinstance(cycle, int):
        timestamp = df.index
        num_to_add = -len(timestamp) % cycle
        # if we want align by the first value, then we should append NaNs to timestamp
        to_add_index = None
        if SeasonalPlotAlignment(alignment) == SeasonalPlotAlignment.first:
            to_add_index = pd.date_range(start=timestamp.max(), periods=num_to_add + 1, closed="right", freq=freq)
        # if we want to align by the last value, then we should prepend NaNs to timestamp
        elif SeasonalPlotAlignment(alignment) == SeasonalPlotAlignment.last:
            to_add_index = pd.date_range(end=timestamp.min(), periods=num_to_add + 1, closed="left", freq=freq)

        df = pd.concat((df, pd.DataFrame(None, index=to_add_index))).sort_index()

    return df
