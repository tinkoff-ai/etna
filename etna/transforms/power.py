from typing import List
from typing import Optional

from sklearn.preprocessing import PowerTransformer

from etna.transforms.sklearn import SklearnTransform


class YeoJohnsonTransform(SklearnTransform):
    """YeoJohnsonTransform applies Yeo-Johns transformation to a DataFrame."""

    def __init__(self, in_columns: Optional[List[str]] = None, inplace: bool = True, standardize: bool = True):
        self.standardize = standardize
        super().__init__(
            in_columns=in_columns,
            inplace=inplace,
            transformer=PowerTransformer(method="yeo-johnson", standardize=self.standardize),
        )


class BoxCoxTransform(SklearnTransform):
    """BoxCoxTransform applies Box-Cox transformation to DataFrame."""

    def __init__(self, in_columns: Optional[List[str]] = None, inplace: bool = True, standardize: bool = True):
        self.standardize = standardize
        super().__init__(
            in_columns=in_columns,
            inplace=inplace,
            transformer=PowerTransformer(method="box-cox", standardize=self.standardize),
        )


__all__ = ["BoxCoxTransform", "YeoJohnsonTransform"]
