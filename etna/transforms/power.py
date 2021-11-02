import warnings
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
        out_column: Optional[str] = None,
        standardize: bool = True,
        mode: Union[TransformMode, str] = "per-segment",
    ):
        """
        Create instance of YeoJohnsonTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        inplace:
            if True, apply transformation inplace to in_column,
            if False, add column to dataset.
        out_column:
            name of added column. Use self.__repr__() if not given
        standardize:
            Set to True to apply zero-mean, unit-variance normalization to the
            transformed output.
        """
        if inplace and (out_column is not None):
            warnings.warn("Transformation will be applied inplace, out_column param will be ignored")
        self.standardize = standardize
        self.inplace = inplace
        self.out_column = out_column
        self.mode = TransformMode(mode)
        self.in_column = [in_column] if isinstance(in_column, str) else in_column
        super().__init__(
            in_column=in_column,
            inplace=inplace,
            out_column=self.out_column if self.out_column is not None else self.__repr__(),
            transformer=PowerTransformer(method="yeo-johnson", standardize=self.standardize),
            mode=mode,
        )


class BoxCoxTransform(SklearnTransform):
    """BoxCoxTransform applies Box-Cox transformation to DataFrame."""

    def __init__(
        self,
        in_column: Optional[Union[str, List[str]]] = None,
        inplace: bool = True,
        out_column: Optional[str] = None,
        standardize: bool = True,
        mode: Union[TransformMode, str] = "per-segment",
    ):
        """
        Create instance of BoxCoxTransform.

        Parameters
        ----------
        in_column:
            name of processed column
        inplace:
            if True, apply transformation inplace to in_column,
            if False, add column to dataset.
        out_column:
            name of added column. Use self.__repr__() if not given.
        standardize:
            Set to True to apply zero-mean, unit-variance normalization to the
            transformed output.
        """
        if inplace and (out_column is not None):
            warnings.warn("Transformation will be applied inplace, out_column param will be ignored")
        self.standardize = standardize
        self.in_column = [in_column] if isinstance(in_column, str) else in_column
        self.inplace = inplace
        self.out_column = out_column
        self.mode = TransformMode(mode)
        super().__init__(
            in_column=in_column,
            inplace=inplace,
            out_column=self.out_column if self.out_column is not None else self.__repr__(),
            transformer=PowerTransformer(method="box-cox", standardize=self.standardize),
            mode=mode,
        )


__all__ = ["BoxCoxTransform", "YeoJohnsonTransform"]
