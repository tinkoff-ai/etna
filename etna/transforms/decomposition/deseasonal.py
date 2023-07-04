import numpy as np
from typing import Dict
from typing import List

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from etna.distributions import BaseDistribution
from etna.distributions import CategoricalDistribution
from etna.transforms.base import OneSegmentTransform
from etna.transforms.base import ReversiblePerSegmentWrapper
from etna.transforms.utils import match_target_quantiles
from etna.models.utils import determine_num_steps
from etna.models.utils import determine_freq


class _OneSegmentDeseasonalityTransform(OneSegmentTransform):
    def __init__(self, in_column: str, period: int, model: str = "additive"):
        """
        Init _OneSegmentDeseasonalityTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        period:
            size of seasonality
        model:
            'additive' (default) or 'multiplicative'
        """
        self.in_column = in_column
        self.period = period

        allowed_models = ("additive", "multiplicative")
        if isinstance(model, str):
            if model in allowed_models:
                self.model = model
            else:
                raise ValueError(f"Not a valid option for model: {model}, only {allowed_models} can be used")

        self.seasonal_ = None

    def _roll_seasonal(self, X: pd.Series) -> np.ndarray:
        """
        Roll out seasonal component by X's time index

        Parameters
        ----------
        X:
            processed column

        Returns
        -------
        result: np.ndarray
            seasonal component
        """
        freq = determine_freq(X.index)
        shift = -determine_num_steps(self.seasonal_.index[0], X.index[0], freq) % self.period
        return np.resize(np.roll(self.seasonal_, shift=shift), X.shape[0])

    def fit(self, df: pd.DataFrame) -> "_OneSegmentDeseasonalityTransform":
        """
        Perform seasonal decomposition.

        Parameters
        ----------
        df:
            Features dataframe with time

        Returns
        -------
        result: _OneSegmentDeseasonalityTransform
            instance after processing
        """
        df = df.loc[: df[self.in_column].last_valid_index()]
        if df[self.in_column].isnull().values.any():
            raise ValueError(
                "The input column contains NaNs in the head or in the middle of the series! Try to use the imputer."
            )
        self.seasonal_ = seasonal_decompose(
            x=df[self.in_column],
            model=self.model,
            filt=None,
            two_sided=False,
            extrapolate_trend=0
        ).seasonal[:self.period]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Subtract seasonal component.

        Parameters
        ----------
        df:
            Features dataframe with time

        Returns
        -------
        result: pd.DataFrame
            Dataframe with extracted features
        """
        result = df
        if self.seasonal_ is not None:
            seasonal = self._roll_seasonal(result[self.in_column])
            if self.model == "additive":
                result[self.in_column] -= seasonal
            else:
                result[self.in_column] /= seasonal
        else:
            raise ValueError("Transform is not fitted! Fit the Transform before calling transform method.")
        return result

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add seasonal component.

        Parameters
        ----------
        df:
            Features dataframe with time

        Returns
        -------
        result: pd.DataFrame
            Dataframe with extracted features
        """
        result = df
        if self.seasonal_ is None:
            raise ValueError("Transform is not fitted! Fit the Transform before calling inverse_transform method.")
        else:
            seasonal = self._roll_seasonal(result[self.in_column])
            if self.model == "additive":
                result[self.in_column] += seasonal
            else:
                result[self.in_column] *= seasonal
        if self.in_column == "target":
            quantiles = match_target_quantiles(set(result.columns))
            for quantile_column_nm in quantiles:
                if self.model == "additive":
                    result.loc[:, quantile_column_nm] += seasonal
                else:
                    result.loc[:, quantile_column_nm] *= seasonal
        return result


class DeseasonalityTransform(ReversiblePerSegmentWrapper):
    """Transform that uses :py:class:`statsmodels.tsa.seasonal.seasonal_decompose` to subtract season from the data.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

    def __init__(self, in_column: str, period: int, model: str = "additive"):
        """
        Init DeseasonalityTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        period:
            size of seasonality
        model:
            'additive' (default) or 'multiplicative'
        """
        self.in_column = in_column
        self.period = period
        self.model = model
        super().__init__(
            transform=_OneSegmentDeseasonalityTransform(
                in_column=self.in_column,
                period=self.period,
                model=self.model,
            ),
            required_features=[self.in_column]
        )

    def get_regressors_info(self) -> List[str]:
        """Return the list with regressors created by the transform."""
        return []

    def params_to_tune(self) -> Dict[str, BaseDistribution]:
        """Get default grid for tuning hyperparameters.

        This grid tunes parameters: ``model``. Other parameters are expected to be set by the user.

        Returns
        -------
        :
            Grid to tune.
        """
        return {
            "model": CategoricalDistribution(["additive", "multiplicative"])
        }
