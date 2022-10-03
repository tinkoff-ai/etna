import math
from typing import Optional
from typing import Sequence

import numpy as np
import pandas as pd

from etna.transforms.base import FutureMixin
from etna.transforms.base import Transform


class FourierTransform(Transform, FutureMixin):
    """Adds fourier features to the dataset.

    Notes
    -----
    To understand how transform works we recommend:
    `Fourier series <https://otexts.com/fpp2/useful-predictors.html#fourier-series>`_.

    * Parameter ``period`` is responsible for the seasonality we want to capture.
    * Parameters ``order`` and ``mods`` define which harmonics will be used.

    Parameter ``order`` is a more user-friendly version of ``mods``.
    For example, ``order=2`` can be represented as ``mods=[1, 2, 3, 4]`` if ``period`` > 4 and
    as ``mods=[1, 2, 3]`` if 3 <= ``period`` <= 4.
    """

    def __init__(
        self,
        period: float,
        order: Optional[int] = None,
        mods: Optional[Sequence[int]] = None,
        out_column: Optional[str] = None,
    ):
        """Create instance of FourierTransform.

        Parameters
        ----------
        period:
            the period of the seasonality to capture in frequency units of time series;

            ``period`` should be >= 2
        order:
            upper order of Fourier components to include;

            ``order`` should be >= 1 and <= ceil(period/2))
        mods:
            alternative and precise way of defining which harmonics will be used,
            for example ``mods=[1, 3, 4]`` means that sin of the first order
            and sin and cos of the second order will be used;

            ``mods`` should be >= 1 and < period
        out_column:

            * if set, name of added column, the final name will be '{out_columnt}_{mod}';

            * if don't set, name will be ``transform.__repr__()``,
              repr will be made for transform that creates exactly this column

        Raises
        ------
        ValueError:
            if period < 2
        ValueError:
            if both or none of order, mods is set
        ValueError:
            if order is < 1 or > ceil(period/2)
        ValueError:
            if at least one mod is < 1 or >= period
        """
        if period < 2:
            raise ValueError("Period should be at least 2")
        self.period = period
        self.mods: Sequence[int]

        if order is not None and mods is None:
            if order < 1 or order > math.ceil(period / 2):
                raise ValueError("Order should be within [1, ceil(period/2)] range")
            self.mods = [mod for mod in range(1, 2 * order + 1) if mod < period]
        elif mods is not None and order is None:
            if min(mods) < 1 or max(mods) >= period:
                raise ValueError("Every mod should be within [1, int(period)) range")
            self.mods = mods
        else:
            raise ValueError("There should be exactly one option set: order or mods")
        self.order = None
        self.out_column = out_column

    def fit(self, df: pd.DataFrame) -> "FourierTransform":
        """Fit method does nothing and is kept for compatibility.

        Parameters
        ----------
        df:
            dataframe with data.

        Returns
        -------
        result: FourierTransform
        """
        return self

    def _get_column_name(self, mod: int) -> str:
        if self.out_column is None:
            return f"{FourierTransform(period=self.period, mods=[mod]).__repr__()}"
        else:
            return f"{self.out_column}_{mod}"

    @staticmethod
    def _construct_answer(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        dataframes = []
        for seg in df.columns.get_level_values("segment").unique():
            tmp = df[seg].join(features)
            _idx = tmp.columns.to_frame()
            _idx.insert(0, "segment", seg)
            tmp.columns = pd.MultiIndex.from_frame(_idx)
            dataframes.append(tmp)

        result = pd.concat(dataframes, axis=1).sort_index(axis=1)
        result.columns.names = ["segment", "feature"]
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add harmonics to the dataset.

        Parameters
        ----------
        df:
            dataframe with data to transform.

        Returns
        -------
        result: pd.Dataframe
            transformed dataframe
        """
        features = pd.DataFrame(index=df.index)
        elapsed = np.arange(features.shape[0]) / self.period

        for mod in self.mods:
            order = (mod + 1) // 2
            is_cos = mod % 2 == 0

            features[self._get_column_name(mod)] = np.sin(2 * np.pi * order * elapsed + np.pi / 2 * is_cos)

        return self._construct_answer(df, features)
