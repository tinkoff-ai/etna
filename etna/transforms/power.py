from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

from sklearn.preprocessing import PowerTransformer

from etna.loggers.base import Logger
from etna.loggers.base import LoggerComposite
from etna.transforms.sklearn import SklearnTransform


class YeoJohnsonTransform(SklearnTransform):
    """YeoJohnsonTransform applies Yeo-Johns transformation to a DataFrame."""

    def __init__(
        self,
        in_column: Optional[Union[str, List[str]]] = None,
        inplace: bool = True,
        standardize: bool = True,
        logger: Union[Logger, Iterable[Logger]] = LoggerComposite(),
    ):
        self.standardize = standardize
        super().__init__(
            in_column=in_column,
            inplace=inplace,
            transformer=PowerTransformer(method="yeo-johnson", standardize=self.standardize),
            logger=logger,
        )


class BoxCoxTransform(SklearnTransform):
    """BoxCoxTransform applies Box-Cox transformation to DataFrame."""

    def __init__(
        self,
        in_column: Optional[Union[str, List[str]]] = None,
        inplace: bool = True,
        standardize: bool = True,
        logger: Union[Logger, Iterable[Logger]] = LoggerComposite(),
    ):
        self.standardize = standardize
        super().__init__(
            in_column=in_column,
            inplace=inplace,
            transformer=PowerTransformer(method="box-cox", standardize=self.standardize),
            logger=logger,
        )


__all__ = ["BoxCoxTransform", "YeoJohnsonTransform"]
