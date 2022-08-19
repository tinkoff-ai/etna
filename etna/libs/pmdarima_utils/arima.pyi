from typing import Any

from numpy import ndarray

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
