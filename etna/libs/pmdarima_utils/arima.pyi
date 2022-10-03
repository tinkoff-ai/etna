from typing import Any

from numpy import ndarray

def check_endog(
    y: Any,
    dtype: Any,
    copy: bool,
    force_all_finite: bool,
    ) -> ndarray: ...

def ARMAtoMA(
    ar: ndarray, 
    ma: ndarray, 
    max_deg: int,
    ) -> ndarray: ...

def seasonal_prediction_with_confidence(
    arima_res: Any, 
    start: Any, 
    end: Any, 
    X: Any, 
    alpha: Any, 
    **kwargs: Any,
    ) -> Any: ...
