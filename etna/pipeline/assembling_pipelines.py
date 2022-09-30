from copy import deepcopy
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from etna.models.base import ModelType
from etna.pipeline.pipeline import Pipeline
from etna.transforms import Transform


def assemble_pipelines(
    models: Union[ModelType, Sequence[ModelType]],
    transforms: Sequence[Union[Transform, Sequence[Optional[Transform]]]],
    horizons: Union[int, Sequence[int]],
) -> List[Pipeline]:
    """Create pipelines with broadcasting from models, transforms and horizons.

    After broadcasting we have:

    - models:
    .. math:: M_1, \dots, M_n
    - transforms:
    .. math:: (T_{1,1}, \dots, T_{1,n}), ... (T_{k,1}, \dots, T_{k,n})
    - horizons:
    .. math:: H_1, \dots, H_n

    We expect that in input shape of size `n` can be reduced to size 1 or even become a scalar value. During broadcasting we copy this value `n` times.

    Parameters
    ----------
    models:
        Instance of Sequence of models
    transforms:
        Sequence of the transforms
    horizons:
        Sequence of horizons

    Returns
    -------
    :
        list of pipelines

    Raises
    ------
    ValueError:
        If the length of models sequence not equals to length of horizons sequence.

    Examples
    --------
    >>> from etna.pipeline import assemble_pipelines
    >>> from etna.models import LinearPerSegmentModel, NaiveModel
    >>> from etna.transforms import TrendTransform, AddConstTransform, LagTransform
    >>> assemble_pipelines(models=LinearPerSegmentModel(), transforms=[LagTransform(in_column='target', lags=[1]), AddConstTransform(in_column='target', value=1)], horizons=[1,2,3])
    [Pipeline(model = LinearPerSegmentModel(fit_intercept = True, ), transforms = [LagTransform(in_column = 'target', lags = [1], out_column = None, ), AddConstTransform(in_column = 'target', value = 1, inplace = True, out_column = None, )], horizon = 1, ),
    Pipeline(model = LinearPerSegmentModel(fit_intercept = True, ), transforms = [LagTransform(in_column = 'target', lags = [1], out_column = None, ), AddConstTransform(in_column = 'target', value = 1, inplace = True, out_column = None, )], horizon = 2, ),
    Pipeline(model = LinearPerSegmentModel(fit_intercept = True, ), transforms = [LagTransform(in_column = 'target', lags = [1], out_column = None, ), AddConstTransform(in_column = 'target', value = 1, inplace = True, out_column = None, )], horizon = 3, )]
    >>> assemble_pipelines(models=[LinearPerSegmentModel(), NaiveModel()], transforms=[LagTransform(in_column='target', lags=[1]), [AddConstTransform(in_column='target', value=1), TrendTransform(in_column='target')]], horizons=[1,2])
    [Pipeline(model = LinearPerSegmentModel(fit_intercept = True, ), transforms = [LagTransform(in_column = 'target', lags = [1], out_column = None, ), AddConstTransform(in_column = 'target', value = 1, inplace = True, out_column = None, )], horizon = 1, ),
    Pipeline(model = NaiveModel(lag = 1, ), transforms = [LagTransform(in_column = 'target', lags = [1], out_column = None, ), TrendTransform(in_column = 'target', out_column = None, detrend_model = LinearRegression(), model = 'ar', custom_cost = None, min_size = 2, jump = 1, n_bkps = 5, pen = None, epsilon = None, )], horizon = 2, )]
    """
    n_models = len(models) if isinstance(models, Sequence) else 1
    n_horizons = len(horizons) if isinstance(horizons, Sequence) else 1
    n_transforms = 1

    for transform_item in transforms:
        if isinstance(transform_item, Sequence):
            if n_transforms != 1 and len(transform_item) != n_transforms:
                raise ValueError(
                    "Transforms elements should be either one Transform, ether sequence of Transforms with same length"
                )
            n_transforms = len(transform_item)

    lengths = {n_models, n_horizons, n_transforms}
    n_pipelines = max(n_models, n_horizons, n_transforms)
    if not len(lengths) == 1 and not (len(lengths) == 2 and 1 in lengths):
        if n_models != 1 and n_models != n_pipelines:
            raise ValueError("Lengths of the result models is not equals to horizons or transforms")
        if n_transforms != 1 and n_transforms != n_pipelines:
            raise ValueError("Lengths of the result transforms is not equals to models or horizons")
        if n_horizons != 1 and n_horizons != n_pipelines:
            raise ValueError("Lengths of the result horizons is not equals to models or transforms")

    models = models if isinstance(models, Sequence) else [models for _ in range(n_pipelines)]
    horizons = horizons if isinstance(horizons, Sequence) else [horizons for _ in range(n_pipelines)]
    transfoms_pipelines: List[List[Any]] = []

    for i in range(n_pipelines):
        transfoms_pipelines.append([])
        for transform in transforms:
            if isinstance(transform, Sequence) and transform[i] is not None:
                transfoms_pipelines[-1].append(transform[i])
            elif isinstance(transform, Transform) and transform is not None:
                transfoms_pipelines[-1].append(transform)

    return [
        Pipeline(deepcopy(model), deepcopy(transform), horizon)
        for model, transform, horizon in zip(models, transfoms_pipelines, horizons)
    ]
