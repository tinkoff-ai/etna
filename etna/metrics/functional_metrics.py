from functools import partial
from typing import List
from typing import Union

import numpy as np
from sklearn.metrics import mean_squared_error as mse

ArrayLike = List[Union[float, List[float]]]


def mape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-15) -> float:
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

    eps: float=1e-15
        MAPE is undefined for ``y_true[i]==0`` for any ``i``, so all zeros ``y_true[i]`` are
        clipped to ``max(eps, abs(y_true))``.

    Returns
    -------
    float
        A non-negative floating point value (the best value is 0.0).
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    y_true_array = y_true_array.clip(eps)

    return np.mean(np.abs((y_true_array - y_pred_array) / y_true_array)) * 100


def smape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-15) -> float:
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

    Returns
    -------
    float
        A non-negative floating point value (the best value is 0.0).
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    return 100 * np.mean(
        2 * np.abs(y_pred_array - y_true_array) / (np.abs(y_true_array) + np.abs(y_pred_array)).clip(eps)
    )


def sign(y_true: ArrayLike, y_pred: ArrayLike) -> float:
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

    Returns
    -------
    float
        A floating point value (the best value is 0.0).
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    return np.mean(np.sign(y_true_array - y_pred_array))


def max_deviation(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Max Deviation metric.

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)

        Estimated target values.

    Returns
    -------
    float
        A floating point value (the best value is 0.0).
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    prefix_error_sum = np.cumsum(y_pred_array - y_true_array)

    return max(np.abs(prefix_error_sum))


rmse = partial(mse, squared=False)


def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
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

    Returns
    -------
    float
        A floating point value (the best value is 0.0).
    """
    y_true_array, y_pred_array = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true_array.shape) != len(y_pred_array.shape):
        raise ValueError("Shapes of the labels must be the same")

    return np.sum(np.abs(y_true_array - y_pred_array)) / np.sum(np.abs(y_true_array))
