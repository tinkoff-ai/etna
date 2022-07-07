from typing import Any
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
    n_models = len(models) if isinstance(models, Sequence) else 1
    n_horizons = len(horizons) if isinstance(horizons, Sequence) else 1
    n_transforms = 1

    for transform in transforms:
        if isinstance(transform, Sequence):
            if n_transforms != 1 and len(transform) != n_transforms:
                raise ValueError("Lengths of the result models, horizons and transforms are not equals")
            n_transforms = len(transform)

    different_lens = {n_models, n_horizons, n_transforms}

    if len(different_lens) > 2 or (len(different_lens) == 2 and 1 not in different_lens):
        raise ValueError("Lengths of the result models, horizons and transforms are not equals")

    models = models if isinstance(models, Sequence) else [models for _ in range(n_transforms)]
    horizons = horizons if isinstance(horizons, Sequence) else [horizons for _ in range(n_transforms)]
    transfoms_pipelines: List[List[Any]] = []

    for i in range(n_transforms):
        transfoms_pipelines.append([])
        for transform in transforms:
            if isinstance(transform, Sequence) and transform[i] is not None:
                transfoms_pipelines[-1].append(transform[i])
            elif isinstance(transform, Transform) and transform is not None:
                transfoms_pipelines[-1].append(transform)
    return [
        Pipeline(model, transform, horizon) for model, transform, horizon in zip(models, transfoms_pipelines, horizons)
    ]
