from typing import List
from typing import Union

import numpy as np

ArrayLike = List[Union[float, List[float]]]


def mape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-15) -> float:
    """Mean absolute percentage error.

    Parameters
    ----------
    y_true:
        array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred:
        array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    eps: float=1e-15
        MAPE is undefined for y_true[i]==0 for any i, so all zeros y_true[i] are
        clipped to max(eps, abs(y_true)).

    Returns
    -------
    float
        A non-negative floating point value (the best value is 0.0).

    References
    ----------
    `Wikipedia entry on the Mean absolute percentage error
    <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`_

    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true.shape) != len(y_pred.shape):
        raise ValueError("Shapes of the labels must be the same")

    y_true = y_true.clip(eps)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def smape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-15) -> float:
    """Symmetric mean absolute percentage error.

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
        MAPE is undefined for y_true[i]==0 for any i, so all zeros y_true[i] are
        clipped to max(eps, abs(y_true)).

    Returns
    -------
    float
        A non-negative floating point value (the best value is 0.0).

    References
    ----------
    `Wikipedia entry on the Symmetric mean absolute percentage error
    <https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error>`_

    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    if len(y_true.shape) != len(y_pred.shape):
        raise ValueError("Shapes of the labels must be the same")

    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)).clip(eps))
