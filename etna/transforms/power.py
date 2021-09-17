from typing import List
from typing import Optional
from typing import Union

from sklearn.preprocessing import PowerTransformer

from etna.transforms.sklearn import SklearnTransform
from etna.transforms.sklearn import TransformMode


class YeoJohnsonTransform(SklearnTransform):
    """YeoJohnsonTransform applies Yeo-Johns transformation to a DataFrame."""

    def __init__(
        self,
        in_column: Optional[Union[str, List[str]]] = None,
        inplace: bool = True,
        standardize: bool = True,
        mode: TransformMode = "per-segment",
    ):
        """
        Create instance of YeoJohnsonTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        inplace:
            if True, apply transformation inplace to in_column,
            if False, add column {yeojohnsontransform}_{in_column} to dataset.
        standardize:
            Set to True to apply zero-mean, unit-variance normalization to the
            transformed output.
        """
        self.standardize = standardize
        super().__init__(
            in_column=in_column,
            inplace=inplace,
            transformer=PowerTransformer(method="yeo-johnson", standardize=self.standardize),
            mode=mode,
        )


class BoxCoxTransform(SklearnTransform):
    """BoxCoxTransform applies Box-Cox transformation to DataFrame."""

    def __init__(
        self,
        in_column: Optional[Union[str, List[str]]] = None,
        inplace: bool = True,
        standardize: bool = True,
        mode: TransformMode = "per-segment",
    ):
        """
        Create instance of BoxCoxTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        inplace:
            if True, apply transformation inplace to in_column,
            if False, add column {boxcoxtransform}_{in_column} to dataset.
        standardize:
            Set to True to apply zero-mean, unit-variance normalization to the
            transformed output.
        """
        self.standardize = standardize
        super().__init__(
            in_column=in_column,
            inplace=inplace,
            transformer=PowerTransformer(method="box-cox", standardize=self.standardize),
            mode=mode,
        )


__all__ = ["BoxCoxTransform", "YeoJohnsonTransform"]
