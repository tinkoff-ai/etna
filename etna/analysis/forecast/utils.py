import warnings
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Union

import pandas as pd

if TYPE_CHECKING:
    from etna.datasets import TSDataset


def get_residuals(forecast_df: pd.DataFrame, ts: "TSDataset") -> "TSDataset":
    """Get residuals for further analysis.

    Function keeps hierarchy, features in result dataset and removes target components.

    Parameters
    ----------
    forecast_df:
        forecasted dataframe with timeseries data
    ts:
        dataset of timeseries that has answers to forecast

    Returns
    -------
    new_ts:
        TSDataset with residuals in forecasts

    Raises
    ------
    KeyError:
        if segments of ``forecast_df`` and ``ts`` aren't the same
    """
    from etna.datasets import TSDataset

    # remove target components
    ts_copy = deepcopy(ts)
    ts_copy.drop_target_components()

    # find the residuals
    true_df = ts_copy[forecast_df.index, :, :]
    if set(ts_copy.segments) != set(forecast_df.columns.get_level_values("segment").unique()):
        raise KeyError("Segments of `ts` and `forecast_df` should be the same")
    true_df.loc[:, pd.IndexSlice[ts.segments, "target"]] -= forecast_df.loc[:, pd.IndexSlice[ts.segments, "target"]]

    # make TSDataset
    new_ts = TSDataset(df=true_df, freq=ts_copy.freq, hierarchical_structure=ts_copy.hierarchical_structure)
    new_ts.known_future = deepcopy(ts_copy.known_future)
    new_ts._regressors = deepcopy(ts_copy.regressors)
    if ts.df_exog is not None:
        new_ts.df_exog = ts.df_exog.copy(deep=True)
    return new_ts


def _get_existing_quantiles(ts: "TSDataset") -> Set[float]:
    """Get quantiles that are present inside the TSDataset."""
    cols = [col for col in ts.columns.get_level_values("feature").unique().tolist() if col.startswith("target_0.")]
    existing_quantiles = {float(col[len("target_") :]) for col in cols}
    return existing_quantiles


def _select_quantiles(forecast_results: Dict[str, "TSDataset"], quantiles: Optional[List[float]]) -> List[float]:
    """Select quantiles from the forecast results.

    Selected quantiles exist in each forecast.
    """
    intersection_quantiles_set = set.intersection(
        *[_get_existing_quantiles(forecast) for forecast in forecast_results.values()]
    )
    intersection_quantiles = sorted(intersection_quantiles_set)

    if quantiles is None:
        selected_quantiles = intersection_quantiles
    else:
        selected_quantiles = sorted(set(quantiles) & intersection_quantiles_set)
        non_existent = set(quantiles) - intersection_quantiles_set
        if non_existent:
            warnings.warn(f"Quantiles {non_existent} do not exist in each forecast dataset. They will be dropped.")

    return selected_quantiles


def _prepare_forecast_results(
    forecast_ts: Union["TSDataset", List["TSDataset"], Dict[str, "TSDataset"]]
) -> Dict[str, "TSDataset"]:
    """Prepare dictionary with forecasts results."""
    from etna.datasets import TSDataset

    if isinstance(forecast_ts, TSDataset):
        return {"1": forecast_ts}
    elif isinstance(forecast_ts, list) and len(forecast_ts) > 0:
        return {str(i + 1): forecast for i, forecast in enumerate(forecast_ts)}
    elif isinstance(forecast_ts, dict) and len(forecast_ts) > 0:
        return forecast_ts
    else:
        raise ValueError("Unknown type of `forecast_ts`")


def _validate_intersecting_segments(fold_numbers: pd.Series):
    """Validate if segments aren't intersecting."""
    fold_info = []
    for fold_number in fold_numbers.unique():
        fold_start = fold_numbers[fold_numbers == fold_number].index.min()
        fold_end = fold_numbers[fold_numbers == fold_number].index.max()
        fold_info.append({"fold_start": fold_start, "fold_end": fold_end})

    fold_info.sort(key=lambda x: x["fold_start"])

    for fold_info_1, fold_info_2 in zip(fold_info[:-1], fold_info[1:]):
        if fold_info_2["fold_start"] <= fold_info_1["fold_end"]:
            raise ValueError("Folds are intersecting")
