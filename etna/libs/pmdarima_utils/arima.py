"""
MIT License

Copyright (c) 2017 Taylor G Smith

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import numpy.polynomial.polynomial as np_polynomial
from pmdarima.utils import diff
from pmdarima.utils import diff_inv
from sklearn.utils.validation import check_array, column_or_1d

DTYPE = np.float64


# Note: Copied from pmdarima package (https://github.com/alkaline-ml/pmdarima/blob/v1.8.5/pmdarima/utils/array.py)
def check_endog(y, dtype=DTYPE, copy=True, force_all_finite=False):
    """Wrapper for ``check_array`` and ``column_or_1d`` from sklearn

    Parameters
    ----------
    y : array-like, shape=(n_samples,)
        The 1d endogenous array.

    dtype : string, type or None (default=np.float64)
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.

    copy : bool, optional (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        still be triggered by a conversion.

    force_all_finite : bool, optional (default=False)
        Whether to raise an error on np.inf and np.nan in an array. The
        possibilities are:

        - True: Force all values of array to be finite.
        - False: accept both np.inf and np.nan in array.

    Returns
    -------
    y : np.ndarray, shape=(n_samples,)
        A 1d numpy ndarray
    """
    return column_or_1d(
        check_array(y, ensure_2d=False, force_all_finite=force_all_finite,
                    copy=copy, dtype=dtype))  # type: np.ndarray


# Note: Copied from pmdarima package (https://github.com/alkaline-ml/pmdarima/blob/v1.8.5/pmdarima/arima/arima.py)
def ARMAtoMA(ar, ma, max_deg):
    r"""
    Convert ARMA coefficients to infinite MA coefficients.
    Compute coefficients of MA model equivalent to given ARMA model.
    MA coefficients are cut off at max_deg.
    The same function as ARMAtoMA() in stats library of R
    Parameters
    ----------
    ar : array-like, shape=(n_orders,)
        The array of AR coefficients.
    ma : array-like, shape=(n_orders,)
        The array of MA coefficients.
    max_deg : int
        Coefficients are computed up to the order of max_deg.
    Returns
    -------
    np.ndarray, shape=(max_deg,)
        Equivalent MA coefficients.
    Notes
    -----
    Here is the derivation. Suppose ARMA model is defined as
    .. math::
    x_t - ar_1*x_{t-1} - ar_2*x_{t-2} - ... - ar_p*x_{t-p}\\
        = e_t + ma_1*e_{t-1} + ma_2*e_{t-2} + ... + ma_q*e_{t-q}
    namely
    .. math::
    (1 - \sum_{i=1}^p[ar_i*B^i]) x_t = (1 + \sum_{i=1}^q[ma_i*B^i]) e_t
    where :math:`B` is a backward operator.
    Equivalent MA model is
    .. math::
        x_t = (1 - \sum_{i=1}^p[ar_i*B^i])^{-1}\\
        * (1 + \sum_{i=1}^q[ma_i*B^i]) e_t\\
        = (1 + \sum_{i=1}[ema_i*B^i]) e_t
    where :math:``ema_i`` is a coefficient of equivalent MA model.
    The :math:``ema_i`` satisfies
    .. math::
        (1 - \sum_{i=1}^p[ar_i*B^i]) * (1 + \sum_{i=1}[ema_i*B^i]) \\
        = 1 + \sum_{i=1}^q[ma_i*B^i]
    thus
    .. math::
        \sum_{i=1}[ema_i*B^i] = \sum_{i=1}^p[ar_i*B^i] \\
        + \sum_{i=1}^p[ar_i*B^i] * \sum_{j=1}[ema_j*B^j] \\
        + \Sum_{i=1}^q[ma_i*B^i]
    therefore
    .. math::
        ema_i = ar_i (but 0 if i>p) \\
        + \Sum_{j=1}^{min(i-1,p)}[ar_j*ema_{i-j}] + ma_i(but 0 if i>q) \\
        = \sum_{j=1}{min(i,p)}[ar_j*ema_{i-j}(but 1 if j=i)] \\
        + ma_i(but 0 if i>q)
    """
    p = len(ar)
    q = len(ma)
    ema = np.empty(max_deg)
    for i in range(0, max_deg):
        temp = ma[i] if i < q else 0.0
        for j in range(0, min(i + 1, p)):
            temp += ar[j] * (ema[i - j - 1] if i - j - 1 >= 0 else 1.0)
        ema[i] = temp
    return ema


# Note: Copied from pmdarima package (https://github.com/alkaline-ml/pmdarima/blob/v1.8.5/pmdarima/arima/arima.py)
def seasonal_prediction_with_confidence(arima_res,
                                        start,
                                        end,
                                        X,
                                        alpha,
                                        **kwargs):
    """Compute the prediction for a SARIMAX and get a conf interval

    Unfortunately, SARIMAX does not really provide a nice way to get the
    confidence intervals out of the box, so we have to perform the
    ``get_prediction`` code here and unpack the confidence intervals manually.
    """
    results = arima_res.get_prediction(
        start=start,
        end=end,
        exog=X,
        **kwargs)

    f = results.predicted_mean
    conf_int = results.conf_int(alpha=alpha)
    if arima_res.specification['simple_differencing']:
        # If simple_differencing == True, statsmodels.get_prediction returns
        # mid and confidence intervals on differenced time series.
        # We have to invert differencing the mid and confidence intervals
        y_org = arima_res.model.orig_endog
        d = arima_res.model.orig_k_diff
        D = arima_res.model.orig_k_seasonal_diff
        period = arima_res.model.seasonal_periods
        # Forecast mid: undifferencing non-seasonal part
        if d > 0:
            y_sdiff = y_org if D == 0 else diff(y_org, period, D)
            f_temp = np.append(y_sdiff[-d:], f)
            f_temp = diff_inv(f_temp, 1, d)
            f = f_temp[(2 * d):]
        # Forecast mid: undifferencing seasonal part
        if D > 0 and period > 1:
            f_temp = np.append(y_org[-(D * period):], f)
            f_temp = diff_inv(f_temp, period, D)
            f = f_temp[(2 * D * period):]
        # confidence interval
        ar_poly = arima_res.polynomial_reduced_ar
        poly_diff = np_polynomial.polypow(np.array([1., -1.]), d)
        sdiff = np.zeros(period + 1)
        sdiff[0] = 1.
        sdiff[-1] = 1.
        poly_sdiff = np_polynomial.polypow(sdiff, D)
        ar = -np.polymul(ar_poly, np.polymul(poly_diff, poly_sdiff))[1:]
        ma = arima_res.polynomial_reduced_ma[1:]
        n_predMinus1 = end - start
        ema = ARMAtoMA(ar, ma, n_predMinus1)
        sigma2 = arima_res._params_variance[0]
        var = np.cumsum(np.append(1., ema * ema)) * sigma2
        q = results.dist.ppf(1. - alpha / 2, *results.dist_args)
        conf_int[:, 0] = f - q * np.sqrt(var)
        conf_int[:, 1] = f + q * np.sqrt(var)

    return check_endog(f, dtype=None, copy=False), \
        check_array(conf_int, copy=False, dtype=None)
