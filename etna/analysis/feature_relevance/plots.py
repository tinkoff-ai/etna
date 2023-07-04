import warnings
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import seaborn as sns
from typing_extensions import Literal

from etna.analysis.feature_relevance.relevance import RelevanceTable
from etna.analysis.feature_relevance.relevance import StatisticsRelevanceTable
from etna.analysis.feature_relevance.utils import _get_fictitious_relevances
from etna.analysis.feature_selection import AGGREGATION_FN
from etna.analysis.feature_selection import AggregationMode
from etna.analysis.utils import _prepare_axes

if TYPE_CHECKING:
    from etna.datasets import TSDataset


def plot_feature_relevance(
    ts: "TSDataset",
    relevance_table: RelevanceTable,
    normalized: bool = False,
    relevance_aggregation_mode: Union[str, Literal["per-segment"]] = AggregationMode.mean,
    relevance_params: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
    alpha: float = 0.05,
    segments: Optional[List[str]] = None,
    columns_num: int = 2,
    figsize: Tuple[int, int] = (10, 5),
):
    """
    Plot relevance of the features.

    The most important features are at the top, the least important are at the bottom.

    For :py:class:`~etna.analysis.feature_relevance.relevance.StatisticsRelevanceTable` also plot vertical line: transformed significance level.

    * Values that lie to the right of this line have p-value < alpha.

    * And the values that lie to the left have p-value > alpha.

    Parameters
    ----------
    ts:
        TSDataset with timeseries data
    relevance_table:
        method to evaluate the feature relevance;

        * if :py:class:`~etna.analysis.feature_relevance.relevance.StatisticsRelevanceTable` table is used then relevances are normalized p-values

        * if :py:class:`~etna.analysis.feature_relevance.relevance.ModelRelevanceTable` table is used then relevances are importances from some model

    normalized:
        whether obtained relevances should be normalized to sum up to 1
    relevance_aggregation_mode:
        aggregation strategy for obtained feature relevance table;
        all the strategies can be examined
        at :py:class:`~etna.analysis.feature_selection.mrmr_selection.AggregationMode`
    relevance_params:
        additional keyword arguments for the ``__call__`` method of
        :py:class:`~etna.analysis.feature_relevance.relevance.RelevanceTable`
    top_k:
        number of best features to plot, if None plot all the features
    alpha:
        significance level, default alpha = 0.05, only for :py:class:`~etna.analysis.feature_relevance.relevance.StatisticsRelevanceTable`
    segments:
        segments to use
    columns_num:
        if ``relevance_aggregation_mode="per-segment"`` number of columns in subplots, otherwise the value is ignored
    figsize:
        size of the figure per subplot with one segment in inches
    """
    if relevance_params is None:
        relevance_params = {}
    if segments is None:
        segments = sorted(ts.segments)
    border_value = None
    features = list(set(ts.columns.get_level_values("feature")) - {"target"})
    relevance_df = relevance_table(df=ts[:, segments, "target"], df_exog=ts[:, segments, features], **relevance_params)
    if relevance_aggregation_mode == "per-segment":
        _, ax = _prepare_axes(num_plots=len(segments), columns_num=columns_num, figsize=figsize)
        for i, segment in enumerate(segments):
            relevance = relevance_df.loc[segment]
            if isinstance(relevance_table, StatisticsRelevanceTable):
                relevance, border_value = _get_fictitious_relevances(
                    relevance,
                    alpha,
                )
            # warning about NaNs
            if relevance.isna().any():
                na_relevance_features = relevance[relevance.isna()].index.tolist()
                warnings.warn(
                    f"Relevances on segment: {segment} of features: {na_relevance_features} can't be calculated."
                )
            relevance = relevance.sort_values(ascending=False)
            relevance = relevance.dropna()[:top_k]
            if normalized:
                if border_value is not None:
                    border_value = border_value / relevance.sum()
                relevance = relevance / relevance.sum()

            sns.barplot(x=relevance.values, y=relevance.index, orient="h", ax=ax[i])
            if border_value is not None:
                ax[i].axvline(border_value)
            ax[i].set_title(f"Feature relevance: {segment}")

    else:
        relevance_aggregation_fn = AGGREGATION_FN[AggregationMode(relevance_aggregation_mode)]
        relevance = relevance_df.apply(lambda x: relevance_aggregation_fn(x[~x.isna()]))  # type: ignore
        if isinstance(relevance_table, StatisticsRelevanceTable):
            relevance, border_value = _get_fictitious_relevances(
                relevance,
                alpha,
            )
        # warning about NaNs
        if relevance.isna().any():
            na_relevance_features = relevance[relevance.isna()].index.tolist()
            warnings.warn(f"Relevances of features: {na_relevance_features} can't be calculated.")
        # if top_k == None, all the values are selected
        relevance = relevance.sort_values(ascending=False)
        relevance = relevance.dropna()[:top_k]
        if normalized:
            if border_value is not None:
                border_value = border_value / relevance.sum()
            relevance = relevance / relevance.sum()

        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        sns.barplot(x=relevance.values, y=relevance.index, orient="h", ax=ax)
        if border_value is not None:
            ax.axvline(border_value)  # type: ignore
        ax.set_title("Feature relevance")  # type: ignore
        ax.grid()  # type: ignore
