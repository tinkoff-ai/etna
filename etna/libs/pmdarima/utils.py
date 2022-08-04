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
from sklearn.utils.validation import check_array
from pmdarima.arima import ARMAtoMA
from pmdarima.utils import diff
from pmdarima.utils import diff_inv
from pmdarima.utils import check_endog


# Note: Originally copied from pmdarima package (https://github.com/blue-yonder/tsfresh/blob/https://github.com/alkaline-ml/pmdarima/blob/v1.8.3/pmdarima/arima/arima.py)
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
