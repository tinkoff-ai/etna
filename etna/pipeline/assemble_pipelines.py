import itertools
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from etna.models.base import BaseModel
from etna.pipeline.pipeline import Pipeline
from etna.transforms import Transform


def assemble_pipelines(
    models: Union[BaseModel, Sequence[BaseModel]],
    transforms: Sequence[Union[Transform, Sequence[Optional[Transform]]]],
    horizons: Union[int, Sequence[int]],
) -> List[Pipeline]:
    """Create pipelines from input horizons and models or sequence of models.

    Parameters
    ----------
    models:
        Instance or Sequence of the etna Model
    transforms:
        Sequence of the transforms
    horizons:
        Sequence of horizons

    Raises
    ------
    ValueError:
        If the length of models sequence not equals to length of horizons sequence.
    Returns
    -------
    list of pipelines
    """
    n_pipelines = 1
    for transform in transforms:
        if isinstance(transform, Sequence):
            n_pipelines *= len(transform)

    if isinstance(models, Sequence) and n_pipelines != len(models):
        raise ValueError("len of models sequence is not equals to expected number of pipelines")

    if isinstance(horizons, Sequence) and n_pipelines != len(horizons):
        raise ValueError("len of horizons sequence is not equals to expected number of pipelines")

    models = models if isinstance(models, Sequence) else [models for _ in range(n_pipelines)]
    horizons = horizons if isinstance(horizons, Sequence) else [horizons for _ in range(n_pipelines)]
    transfoms_seq = [transform if isinstance(transform, Sequence) else [transform] for transform in transforms]
    transforms_prod = list(itertools.product(*transfoms_seq))
    transforms_without_nans = [[elem for elem in transform if elem is not None] for transform in transforms_prod]

    return [
        Pipeline(model, transform, horizon)
        for model, transform, horizon in zip(models, transforms_without_nans, horizons)
    ]
