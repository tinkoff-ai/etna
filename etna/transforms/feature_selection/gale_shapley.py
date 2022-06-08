import warnings
from math import ceil
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
from typing_extensions import Literal

from etna.analysis import RelevanceTable
from etna.core import BaseMixin
from etna.transforms.feature_selection.base import BaseFeatureSelectionTransform


class BaseGaleShapley(BaseMixin):
    """Base class for a member of Gale-Shapley matching."""

    def __init__(self, name: str, ranked_candidates: List[str]):
        """Init BaseGaleShapley.

        Parameters
        ----------
        name:
            name of object
        ranked_candidates:
            list of preferences for the object ranked descending by importance
        """
        self.name = name
        self.ranked_candidate = ranked_candidates
        self.candidates_rank = {candidate: i for i, candidate in enumerate(self.ranked_candidate)}
        self.tmp_match: Optional[str] = None
        self.tmp_match_rank: Optional[int] = None
        self.is_available = True

    def update_tmp_match(self, name: str):
        """Create match with object name.

        Parameters
        ----------
        name:
            name of candidate to match
        """
        self.tmp_match = name
        self.tmp_match_rank = self.candidates_rank[name]
        self.is_available = False

    def reset_tmp_match(self):
        """Break tmp current."""
        self.tmp_match = None
        self.tmp_match_rank = None
        self.is_available = True


class SegmentGaleShapley(BaseGaleShapley):
    """Class for segment member of Gale-Shapley matching."""

    def __init__(self, name: str, ranked_candidates: List[str]):
        """Init SegmentGaleShapley.

        Parameters
        ----------
        name:
            name of segment
        ranked_candidates:
            list of features sorted descending by importance
        """
        super().__init__(name=name, ranked_candidates=ranked_candidates)
        self.last_candidate: Optional[int] = None

    def update_tmp_match(self, name: str):
        """Create match with given feature.

        Parameters
        ----------
        name:
            name of feature to match
        """
        super().update_tmp_match(name=name)
        self.last_candidate = self.tmp_match_rank

    def get_next_candidate(self) -> Optional[str]:
        """Get name of the next feature to try.

        Returns
        -------
        name: str
            name of feature
        """
        if self.last_candidate is None:
            self.last_candidate = 0
        else:
            self.last_candidate += 1

        if self.last_candidate >= len(self.ranked_candidate):
            return None
        return self.ranked_candidate[self.last_candidate]


class FeatureGaleShapley(BaseGaleShapley):
    """Class for feature member of Gale-Shapley matching."""

    def check_segment(self, segment: str) -> bool:
        """Check if given segment is better than current match according to preference list.

        Parameters
        ----------
        segment:
            segment to check

        Returns
        -------
        is_better: bool
            returns True if given segment is a better candidate than current match.
        """
        if self.tmp_match is None or self.tmp_match_rank is None:
            return True
        return self.candidates_rank[segment] < self.tmp_match_rank


class GaleShapleyMatcher(BaseMixin):
    """Class for handling Gale-Shapley matching algo."""

    def __init__(self, segments: List[SegmentGaleShapley], features: List[FeatureGaleShapley]):
        """Init GaleShapleyMatcher.

        Parameters
        ----------
        segments:
            list of segments to build matches
        features:
            list of features to build matches
        """
        self.segments = segments
        self.features = features
        self.segment_by_name = {segment.name: segment for segment in self.segments}
        self.feature_by_name = {feature.name: feature for feature in self.features}

    @staticmethod
    def match(segment: SegmentGaleShapley, feature: FeatureGaleShapley):
        """Build match between segment and feature.

        Parameters
        ----------
        segment:
            segment to match
        feature:
            feature to match
        """
        segment.update_tmp_match(name=feature.name)
        feature.update_tmp_match(name=segment.name)

    @staticmethod
    def break_match(segment: SegmentGaleShapley, feature: FeatureGaleShapley):
        """Break match between segment and feature.

        Parameters
        ----------
        segment:
            segment to break match
        feature:
            feature to break match
        """
        segment.reset_tmp_match()
        feature.reset_tmp_match()

    def _gale_shapley_iteration(self, available_segments: List[SegmentGaleShapley]) -> bool:
        """
        Run iteration of Gale-Shapley matching for given available_segments.

        Parameters
        ----------
        available_segments:
            list of segments that have no match at this iteration

        Returns
        -------
        success: bool
            True if there is at least one match attempt at the iteration

        Notes
        -----
        Success code is necessary because in ETNA usage we can not guarantee that number of features will be
        big enough to build matches with all the segments. In case ``n_features < n_segments`` some segments always stay
        available that can cause infinite while loop in ``__call__``.
        """
        success = False
        for segment in available_segments:
            next_feature_candidate_name = segment.get_next_candidate()
            if next_feature_candidate_name is None:
                continue
            next_feature_candidate = self.feature_by_name[next_feature_candidate_name]
            success = True
            if next_feature_candidate.check_segment(segment=segment.name):
                if not next_feature_candidate.is_available:  # is_available = tmp_match is not None
                    self.break_match(
                        segment=self.segment_by_name[next_feature_candidate.tmp_match],  # type: ignore
                        feature=next_feature_candidate,
                    )
                self.match(segment=segment, feature=next_feature_candidate)
        return success

    def _get_available_segments(self) -> List[SegmentGaleShapley]:
        """Get list of available segments."""
        return [segment for segment in self.segments if segment.is_available]

    def __call__(self) -> Dict[str, str]:
        """Run matching.

        Returns
        -------
        matching: Dict[str, str]
            matching dict of segment x feature
        """
        success_run = True
        available_segments = self._get_available_segments()
        while available_segments and success_run:
            success_run = self._gale_shapley_iteration(available_segments=available_segments)
            available_segments = self._get_available_segments()
        return {segment.name: segment.tmp_match for segment in self.segments if segment.tmp_match is not None}


class GaleShapleyFeatureSelectionTransform(BaseFeatureSelectionTransform):
    """GaleShapleyFeatureSelectionTransform provides feature filtering with Gale-Shapley matching algo according to relevance table.


    Notes
    -----
    Transform works with any type of features, however most of the models works only with regressors.
    Therefore, it is recommended to pass the regressors into the feature selection transforms.
    """

    def __init__(
        self,
        relevance_table: RelevanceTable,
        top_k: int,
        features_to_use: Union[List[str], Literal["all"]] = "all",
        use_rank: bool = False,
        return_features: bool = False,
        **relevance_params,
    ):
        """Init GaleShapleyFeatureSelectionTransform.

        Parameters
        ----------
        relevance_table:
            class to build relevance table
        top_k:
            number of features that should be selected from all the given ones
        features_to_use:
            columns of the dataset to select from
            if "all" value is given, all columns are used
        use_rank:
            if True, use rank in relevance table computation
        return_features:
            indicates whether to return features or not.
        """
        super().__init__(features_to_use=features_to_use, return_features=return_features)
        self.relevance_table = relevance_table
        self.top_k = top_k
        self.use_rank = use_rank
        self.greater_is_better = False if use_rank else relevance_table.greater_is_better
        self.relevance_params = relevance_params

    def _compute_relevance_table(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Compute relevance table with given data."""
        targets_df = df.loc[:, pd.IndexSlice[:, "target"]]
        features_df = df.loc[:, pd.IndexSlice[:, features]]
        table = self.relevance_table(
            df=targets_df, df_exog=features_df, return_ranks=self.use_rank, **self.relevance_params
        )
        return table

    @staticmethod
    def _get_ranked_list(table: pd.DataFrame, ascending: bool) -> Dict[str, List[str]]:
        """Get ranked lists of candidates from table of relevance."""
        ranked_features = {key: list(table.loc[key].sort_values(ascending=ascending).index) for key in table.index}
        return ranked_features

    @staticmethod
    def _compute_gale_shapley_steps_number(top_k: int, n_segments: int, n_features: int) -> int:
        """Get number of necessary Gale-Shapley algo iterations."""
        if n_features < top_k:
            warnings.warn(
                f"Given top_k={top_k} is bigger than n_features={n_features}. " f"Transform will not filter features."
            )
            return 1
        if top_k < n_segments:
            warnings.warn(
                f"Given top_k={top_k} is less than n_segments. Algo will filter data without Gale-Shapley run."
            )
            return 1
        return ceil(top_k / n_segments)

    @staticmethod
    def _gale_shapley_iteration(
        segment_features_ranking: Dict[str, List[str]],
        feature_segments_ranking: Dict[str, List[str]],
    ) -> Dict[str, str]:
        """Build matching for all the segments.

        Parameters
        ----------
        segment_features_ranking:
            dict of relevance segment x sorted features

        Returns
        -------
        matching dict: Dict[str, str]
            dict of segment x feature
        """
        gssegments = [
            SegmentGaleShapley(
                name=name,
                ranked_candidates=ranked_candidates,
            )
            for name, ranked_candidates in segment_features_ranking.items()
        ]
        gsfeatures = [
            FeatureGaleShapley(name=name, ranked_candidates=ranked_candidates)
            for name, ranked_candidates in feature_segments_ranking.items()
        ]
        matcher = GaleShapleyMatcher(segments=gssegments, features=gsfeatures)
        new_matches = matcher()
        return new_matches

    @staticmethod
    def _update_ranking_list(
        segment_features_ranking: Dict[str, List[str]], features_to_drop: List[str]
    ) -> Dict[str, List[str]]:
        """Delete chosen features from candidates ranked lists."""
        for segment in segment_features_ranking:
            for feature in features_to_drop:
                segment_features_ranking[segment].remove(feature)
        return segment_features_ranking

    @staticmethod
    def _process_last_step(
        matches: Dict[str, str], relevance_table: pd.DataFrame, n: int, greater_is_better: bool
    ) -> List[str]:
        """Choose n features from given ones according to relevance_matrix."""
        features_relevance = {feature: relevance_table[feature][segment] for segment, feature in matches.items()}
        sorted_features = sorted(features_relevance.items(), key=lambda item: item[1], reverse=greater_is_better)
        selected_features = [feature[0] for feature in sorted_features][:n]
        return selected_features

    def fit(self, df: pd.DataFrame) -> "GaleShapleyFeatureSelectionTransform":
        """Fit Gale-Shapley algo and find a pool of ``top_k`` features.

        Parameters
        ----------
        df:
            dataframe to fit algo
        """
        features = self._get_features_to_use(df=df)
        relevance_table = self._compute_relevance_table(df=df, features=features)
        segment_features_ranking = self._get_ranked_list(
            table=relevance_table, ascending=not self.relevance_table.greater_is_better
        )
        feature_segments_ranking = self._get_ranked_list(
            table=relevance_table.T, ascending=not self.relevance_table.greater_is_better
        )
        gale_shapley_steps_number = self._compute_gale_shapley_steps_number(
            top_k=self.top_k,
            n_segments=len(segment_features_ranking),
            n_features=len(feature_segments_ranking),
        )
        last_step_features_number = self.top_k % len(segment_features_ranking)
        for step in range(gale_shapley_steps_number):
            matches = self._gale_shapley_iteration(
                segment_features_ranking=segment_features_ranking,
                feature_segments_ranking=feature_segments_ranking,
            )
            if step == gale_shapley_steps_number - 1:
                selected_features = self._process_last_step(
                    matches=matches,
                    relevance_table=relevance_table,
                    n=last_step_features_number,
                    greater_is_better=self.greater_is_better,
                )
            else:
                selected_features = list(matches.values())
            self.selected_features.extend(selected_features)
            segment_features_ranking = self._update_ranking_list(
                segment_features_ranking=segment_features_ranking, features_to_drop=selected_features
            )
        return self
