from copy import deepcopy
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pyts.approximation import SymbolicFourierApproximation
from pyts.transformation import WEASEL
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from typing_extensions import Literal

from etna.experimental.classification.feature_extraction.base import BaseTimeSeriesFeatureExtractor
from etna.experimental.classification.utils import padd_single_series


class CustomWEASEL(WEASEL):
    """Improved version of WEASEL transform to work with the series of different length."""

    def __init__(
        self,
        padding_value: Union[float, Literal["back_fill"]],
        word_size: int,
        ngram_range: Tuple[int, int],
        n_bins: int,
        window_sizes: List[Union[float, int]],
        window_steps: Optional[List[Union[float, int]]],
        anova: bool,
        drop_sum: bool,
        norm_mean: bool,
        norm_std: bool,
        strategy: str,
        chi2_threshold: float,
        sparse: bool,
        alphabet: Optional[Union[List[str]]],
    ):
        """Init CustomWEASEL with given parameters.

        Parameters
        ----------
        padding_value:
            Value to pad the series to fit the `series_len`, if equals to "back_fill" the first value in
            the series is used.
        word_size:
            Size of each word.
        ngram_range:
            The lower and upper boundary of the range of ngrams.
        n_bins:
            The number of bins to produce. It must be between 2 and 26.
        window_sizes:
            Size of the sliding windows. All the elements must be either integers
            or floats. In the latter case, each element represents the percentage
            of the size of each time series and must be between 0 and 1; the size
            of the sliding windows will be computed as
            ``np.ceil(window_sizes * n_timestamps)``.
        window_steps:
            Step of the sliding windows. If None, each ``window_step`` is equal to
            ``window_size`` so that the windows are non-overlapping. Otherwise, all
            the elements must be either integers or floats. In the latter case,
            each element represents the percentage of the size of each time series
            and must be between 0 and 1; the step of the sliding windows will be
            computed as ``np.ceil(window_steps * n_timestamps)``.
        anova:
            If True, the Fourier coefficient selection is done via a one-way
            ANOVA test. If False, the first Fourier coefficients are selected.
        drop_sum:
            If True, the first Fourier coefficient (i.e. the sum of the subseries)
            is dropped. Otherwise, it is kept.
        norm_mean:
            If True, center each subseries before scaling.
        norm_std:
            If True, scale each subseries to unit variance.
        strategy:
            Strategy used to define the widths of the bins:
            - 'uniform': All bins in each sample have identical widths
            - 'quantile': All bins in each sample have the same number of points
            - 'normal': Bin edges are quantiles from a standard normal distribution
            - 'entropy': Bin edges are computed using information gain
        chi2_threshold:
            The threshold used to perform feature selection. Only the words with
            a chi2 statistic above this threshold will be kept.
        sparse:
            Return a sparse matrix if True, else return an array.
        alphabet:
            Alphabet to use. If None, the first `n_bins` letters of the Latin
            alphabet are used.
        """
        super().__init__(
            word_size=word_size,
            n_bins=n_bins,
            window_sizes=window_sizes,
            window_steps=window_steps,
            anova=anova,
            drop_sum=drop_sum,
            norm_mean=norm_mean,
            norm_std=norm_std,
            strategy=strategy,
            chi2_threshold=chi2_threshold,
            sparse=sparse,
            alphabet=alphabet,
        )
        self.padding_value = padding_value
        self.ngram_range = ngram_range
        self._min_series_len: Optional[int] = None
        self._sfa_list: List[SymbolicFourierApproximation] = []
        self._vectorizer_list: List[CountVectorizer] = []
        self._relevant_features_list: List[int] = []
        self._vocabulary: Dict[int, str] = {}
        self._sfa = SymbolicFourierApproximation(
            n_coefs=self.word_size,
            drop_sum=self.drop_sum,
            anova=self.anova,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            n_bins=self.n_bins,
            strategy=self.strategy,
            alphabet=self.alphabet,
        )
        self._padding_expected_len: Optional[int] = None

    @staticmethod
    def _windowed_view(
        x: List[np.ndarray], y: Optional[np.ndarray], window_size: int, window_step: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """Create the samples of length window_size with window_step."""
        n_samples = len(x)
        n_windows_per_sample = [((len(x[i]) - window_size + window_step) // window_step) for i in range(n_samples)]
        n_windows_per_sample_cum = np.asarray(np.concatenate(([0], np.cumsum(n_windows_per_sample))))
        x_windowed = np.asarray(
            np.concatenate(
                [sliding_window_view(series[::-1], window_shape=window_size)[::window_step][::-1, ::-1] for series in x]
            )
        )
        y_windowed = np.asarray(
            y if y is None else np.concatenate([np.repeat(y[i], n_windows_per_sample[i]) for i in range(n_samples)])
        )
        return x_windowed, y_windowed, n_windows_per_sample_cum

    def fit(self, x: List[np.ndarray], y: Optional[np.ndarray] = None) -> "CustomWEASEL":
        """Fit the feature extractor.

        Parameters
        ----------
        x:
            Array with time series.
        y:
            Array of class labels.

        Returns
        -------
        :
            Fitted instance of feature extractor.
        """
        n_samples, self._min_series_len = len(x), np.min(list(map(len, x)))
        window_sizes, window_steps = self._check_params(self._min_series_len)
        self._padding_expected_len = max(window_sizes)

        for (window_size, window_step) in zip(window_sizes, window_steps):
            x_windowed, y_windowed, n_windows_per_sample_cum = self._windowed_view(
                x=x, y=y, window_size=window_size, window_step=window_step
            )

            sfa = deepcopy(self._sfa)
            x_sfa = sfa.fit_transform(x_windowed, y_windowed)
            x_word = np.asarray(["".join(encoded_subseries) for encoded_subseries in x_sfa])
            x_bow = np.asarray(
                [
                    " ".join(x_word[n_windows_per_sample_cum[i] : n_windows_per_sample_cum[i + 1]])
                    for i in range(n_samples)
                ]
            )

            vectorizer = CountVectorizer(ngram_range=self.ngram_range)
            x_counts = vectorizer.fit_transform(x_bow)
            chi2_statistics, _ = chi2(x_counts, y)
            relevant_features = np.where(chi2_statistics > self.chi2_threshold)[0]

            old_length_vocab = len(self._vocabulary)
            vocabulary = {value: key for (key, value) in vectorizer.vocabulary_.items()}
            for i, idx in enumerate(relevant_features):
                self._vocabulary[i + old_length_vocab] = str(window_size) + " " + vocabulary[idx]

            self._relevant_features_list.append(relevant_features)
            self._sfa_list.append(sfa)
            self._vectorizer_list.append(vectorizer)

        return self

    def transform(self, x: List[np.ndarray]) -> np.ndarray:
        """Extract weasel features from the input data.

        Parameters
        ----------
        x:
            Array with time series.

        Returns
        -------
        :
            Transformed input data.
        """
        n_samples = len(x)
        window_sizes, window_steps = self._check_params(self._min_series_len)
        for i in range(len(x)):
            x[i] = (
                x[i]
                if len(x[i]) >= max(window_sizes)
                else padd_single_series(
                    x=x[i], expected_len=self._padding_expected_len, padding_value=self.padding_value
                )
            )

        x_features = coo_matrix((n_samples, 0), dtype=np.int64)

        for (window_size, window_step, sfa, vectorizer, relevant_features) in zip(
            window_sizes, window_steps, self._sfa_list, self._vectorizer_list, self._relevant_features_list
        ):
            x_windowed, _, n_windows_per_sample_cum = self._windowed_view(
                x=x, y=None, window_size=window_size, window_step=window_step
            )
            x_sfa = sfa.transform(x_windowed)
            x_word = np.asarray(["".join(encoded_subseries) for encoded_subseries in x_sfa])
            x_bow = np.asarray(
                [
                    " ".join(x_word[n_windows_per_sample_cum[i] : n_windows_per_sample_cum[i + 1]])
                    for i in range(n_samples)
                ]
            )
            x_counts = vectorizer.transform(x_bow)[:, relevant_features]
            x_features = hstack([x_features, x_counts])

        if not self.sparse:
            return x_features.A
        return csr_matrix(x_features)

    def fit_transform(self, x: List[np.ndarray], y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the feature extractor and extract weasel features from the input data.

        Parameters
        ----------
        x:
            Array with time series.

        Returns
        -------
        :
            Transformed input data.
        """
        return self.fit(x=x, y=y).transform(x=x)


class WEASELFeatureExtractor(BaseTimeSeriesFeatureExtractor):
    """Class to extract features with WEASEL algorithm."""

    def __init__(
        self,
        padding_value: Union[float, Literal["back_fill"]],
        word_size: int = 4,
        ngram_range: Tuple[int, int] = (1, 2),
        n_bins: int = 4,
        window_sizes: Optional[List[Union[float, int]]] = None,
        window_steps: Optional[List[Union[float, int]]] = None,
        anova: bool = True,
        drop_sum: bool = True,
        norm_mean: bool = True,
        norm_std: bool = True,
        strategy: str = "entropy",
        chi2_threshold: float = 2,
        sparse: bool = True,
        alphabet: Optional[Union[List[str]]] = None,
    ):
        """Init WEASELFeatureExtractor with given parameters.

        Parameters
        ----------
        padding_value:
            Value to pad the series to fit the `series_len`, if equals to "back_fill" the first value in
            the series is used.
        word_size:
            Size of each word.
        ngram_range:
            The lower and upper boundary of the range of ngrams.
        n_bins:
            The number of bins to produce. It must be between 2 and 26.
        window_sizes:
            Size of the sliding windows. All the elements must be either integers
            or floats. In the latter case, each element represents the percentage
            of the size of each time series and must be between 0 and 1; the size
            of the sliding windows will be computed as
            ``np.ceil(window_sizes * n_timestamps)``.
        window_steps:
            Step of the sliding windows. If None, each ``window_step`` is equal to
            ``window_size`` so that the windows are non-overlapping. Otherwise, all
            the elements must be either integers or floats. In the latter case,
            each element represents the percentage of the size of each time series
            and must be between 0 and 1; the step of the sliding windows will be
            computed as ``np.ceil(window_steps * n_timestamps)``.
        anova:
            If True, the Fourier coefficient selection is done via a one-way
            ANOVA test. If False, the first Fourier coefficients are selected.
        drop_sum:
            If True, the first Fourier coefficient (i.e. the sum of the subseries)
            is dropped. Otherwise, it is kept.
        norm_mean:
            If True, center each subseries before scaling.
        norm_std:
            If True, scale each subseries to unit variance.
        strategy:
            Strategy used to define the widths of the bins:
            - 'uniform': All bins in each sample have identical widths
            - 'quantile': All bins in each sample have the same number of points
            - 'normal': Bin edges are quantiles from a standard normal distribution
            - 'entropy': Bin edges are computed using information gain
        chi2_threshold:
            The threshold used to perform feature selection. Only the words with
            a chi2 statistic above this threshold will be kept.
        sparse:
            Return a sparse matrix if True, else return an array.
        alphabet:
            Alphabet to use. If None, the first `n_bins` letters of the Latin
            alphabet are used.
        """
        self.weasel = CustomWEASEL(
            padding_value=padding_value,
            word_size=word_size,
            ngram_range=ngram_range,
            n_bins=n_bins,
            window_sizes=window_sizes if window_sizes is not None else [0.1, 0.3, 0.5, 0.7, 0.9],
            window_steps=window_steps,
            anova=anova,
            drop_sum=drop_sum,
            norm_mean=norm_mean,
            norm_std=norm_std,
            strategy=strategy,
            chi2_threshold=chi2_threshold,
            sparse=sparse,
            alphabet=alphabet,
        )

    def fit(self, x: List[np.ndarray], y: Optional[np.ndarray] = None) -> "WEASELFeatureExtractor":
        """Fit the feature extractor.

        Parameters
        ----------
        x:
            Array with time series.
        y:
            Array of class labels.

        Returns
        -------
        :
            Fitted instance of feature extractor.
        """
        self.weasel.fit(x, y)
        return self

    def transform(self, x: List[np.ndarray]) -> np.ndarray:
        """Extract weasel features from the input data.

        Parameters
        ----------
        x:
            Array with time series.

        Returns
        -------
        :
            Transformed input data.
        """
        return self.weasel.transform(x)
