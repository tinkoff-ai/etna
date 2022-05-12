from typing import List
from typing import Optional
from typing import Union

from sklearn.preprocessing import PowerTransformer

from etna.transforms.math.sklearn import SklearnTransform
from etna.transforms.math.sklearn import TransformMode


class YeoJohnsonTransform(SklearnTransform):
    """YeoJohnsonTransform applies Yeo-Johns transformation to a DataFrame.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

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
            columns to be transformed, if None - all columns will be transformed.
        inplace:

            * if True, apply transformation inplace to in_column,

            * if False, add column to dataset.

        out_column:
            base for the names of generated columns, uses ``self.__repr__()`` if not given.
        standardize:
            Set to True to apply zero-mean, unit-variance normalization to the
            transformed output.

        Raises
        ------
        ValueError:
            if incorrect mode given
        """
        self.standardize = standardize
        super().__init__(
            in_column=in_column,
            inplace=inplace,
            out_column=out_column,
            transformer=PowerTransformer(method="yeo-johnson", standardize=self.standardize),
            mode=mode,
        )


class BoxCoxTransform(SklearnTransform):
    """BoxCoxTransform applies Box-Cox transformation to DataFrame.

    Warning
    -------
    This transform can suffer from look-ahead bias. For transforming data at some timestamp
    it uses information from the whole train part.
    """

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
            columns to be transformed, if None - all columns will be transformed.
        inplace:

            * if True, apply transformation inplace to in_column,

            * if False, add column to dataset.

        out_column:
            base for the names of generated columns, uses ``self.__repr__()`` if not given.
        standardize:
            Set to True to apply zero-mean, unit-variance normalization to the
            transformed output.

        Raises
        ------
        ValueError:
            if incorrect mode given
        """
        self.standardize = standardize
        super().__init__(
            in_column=in_column,
            inplace=inplace,
            out_column=out_column,
            transformer=PowerTransformer(method="box-cox", standardize=self.standardize),
            mode=mode,
        )


__all__ = ["BoxCoxTransform", "YeoJohnsonTransform"]
