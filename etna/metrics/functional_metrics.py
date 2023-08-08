from enum import Enum
from functools import partial
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import median_absolute_error as medae
from sklearn.metrics import r2_score
from typing_extensions import assert_never

ArrayLike = Union[float, Sequence[float], Sequence[Sequence[float]]]


class FunctionalMetricMultioutput(str, Enum):
    """Enum for different functional metric multioutput modes."""

    #: Compute one scalar value taking into account all outputs.
    joint = "joint"

    #: Compute one value per each output.
    raw_values = "raw_values"

    @classmethod
    def _missing_(cls, value):
        raise NotImplementedError(
            f"{value} is not a valid {cls.__name__}. Only {', '.join([repr(m.value) for m in cls])} options allowed"
        )


def _get_axis_by_multioutput(multioutput: str) -> Optional[int]:
    multioutput_enum = FunctionalMetricMultioutput(multioutput)
    if multioutput_enum is FunctionalMetricMultioutput.joint:
        return None
    elif multioutput_enum is FunctionalMetricMultioutput.raw_values:
        return 0
    else:
        assert_never(multioutput_enum)


def mape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-15, multioutput: str = "joint") -> ArrayLike:
    """Mean absolute percentage error.

    `Wikipedia entry on the Mean absolute percentage error
    <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`_

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Estimated target values.

    eps:
        MAPE is undefined for ``y_true[i]==0`` for any ``i``, so all zeros ``y_true[i]`` are
        clipped to ``max(eps, abs(y_true))``.

    multioutput:
        Defines aggregating of multiple output values
        (see :py:class:`~etna.metrics.functional_metrics.FunctionalMetricMultioutput`).

    Returns
    -------
    :
        A non-negative floating point value (the best value is 0.0), or an array of floating point values,
        one for each individual target.
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    y_true_array = y_true_array.clip(eps)

    axis = _get_axis_by_multioutput(multioutput)

    return np.mean(np.abs((y_true_array - y_pred_array) / y_true_array), axis=axis) * 100


def smape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-15, multioutput: str = "joint") -> ArrayLike:
    """Symmetric mean absolute percentage error.

    `Wikipedia entry on the Symmetric mean absolute percentage error
    <https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error>`_

    .. math::
        SMAPE = \dfrac{100}{n}\sum_{t=1}^{n}\dfrac{|ytrue_{t}-ypred_{t}|}{(|ypred_{t}|+|ytrue_{t}|) / 2}

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Estimated target values.

    eps: float=1e-15
        SMAPE is undefined for ``y_true[i] + y_pred[i] == 0`` for any ``i``, so all zeros ``y_true[i] + y_pred[i]`` are
        clipped to ``max(eps, abs(y_true) + abs(y_pred))``.

    multioutput:
        Defines aggregating of multiple output values
        (see :py:class:`~etna.metrics.functional_metrics.FunctionalMetricMultioutput`).

    Returns
    -------
    :
        A non-negative floating point value (the best value is 0.0), or an array of floating point values,
        one for each individual target.
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    axis = _get_axis_by_multioutput(multioutput)

    return 100 * np.mean(
        2 * np.abs(y_pred_array - y_true_array) / (np.abs(y_true_array) + np.abs(y_pred_array)).clip(eps), axis=axis
    )


def sign(y_true: ArrayLike, y_pred: ArrayLike, multioutput: str = "joint") -> ArrayLike:
    """Sign error metric.

    .. math::
        Sign(y\_true, y\_pred) = \\frac{1}{n}\\cdot\\sum_{i=0}^{n - 1}{sign(y\_true_i - y\_pred_i)}

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Estimated target values.

    multioutput:
        Defines aggregating of multiple output values
        (see :py:class:`~etna.metrics.functional_metrics.FunctionalMetricMultioutput`).

    Returns
    -------
    :
        A floating point value, or an array of floating point values,
        one for each individual target.
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    axis = _get_axis_by_multioutput(multioutput)

    return np.mean(np.sign(y_true_array - y_pred_array), axis=axis)


def max_deviation(y_true: ArrayLike, y_pred: ArrayLike, multioutput: str = "joint") -> ArrayLike:
    """Max Deviation metric.

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Estimated target values.

    multioutput:
        Defines aggregating of multiple output values
        (see :py:class:`~etna.metrics.functional_metrics.FunctionalMetricMultioutput`).

    Returns
    -------
    :
        A non-negative floating point value (the best value is 0.0), or an array of floating point values,
        one for each individual target.
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    axis = _get_axis_by_multioutput(multioutput)

    prefix_error_sum = np.cumsum(y_pred_array - y_true_array, axis=axis)
    return np.max(np.abs(prefix_error_sum), axis=axis)


rmse = partial(mse, squared=False)


def wape(y_true: ArrayLike, y_pred: ArrayLike, multioutput: str = "joint") -> ArrayLike:
    """Weighted average percentage Error metric.

    .. math::
        WAPE(y\_true, y\_pred) = \\frac{\\sum_{i=0}^{n} |y\_true_i - y\_pred_i|}{\\sum_{i=0}^{n}|y\\_true_i|}

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Estimated target values.

    multioutput:
        Defines aggregating of multiple output values
        (see :py:class:`~etna.metrics.functional_metrics.FunctionalMetricMultioutput`).

    Returns
    -------
    :
        A non-negative floating point value (the best value is 0.0), or an array of floating point values,
        one for each individual target.
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    axis = _get_axis_by_multioutput(multioutput)

    return np.sum(np.abs(y_true_array - y_pred_array), axis=axis) / np.sum(np.abs(y_true_array), axis=axis)  # type: ignore


__all__ = ["mae", "mse", "msle", "medae", "r2_score", "mape", "smape", "sign", "max_deviation", "rmse", "wape"]
