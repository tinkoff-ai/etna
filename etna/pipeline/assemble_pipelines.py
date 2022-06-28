from typing import List
from typing import Sequence
from typing import Union

from etna.models.base import BaseModel
from etna.pipeline.pipeline import Pipeline
from etna.transforms import Transform


def assemble_pipelines(
    models: Union[BaseModel, Sequence[BaseModel]],
    transforms: Sequence[Transform],
    horizons: Sequence[int],
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

    Returns
    -------
    list of pipelines
    """
    if isinstance(models, Sequence):
        if len(models) != len(horizons):
            raise ValueError("len of sequences model and horizons must be equal")
        return [
            Pipeline(model=model, transforms=transforms, horizon=horizon) for model, horizon in zip(models, horizons)
        ]
    else:
        return [Pipeline(model=models, transforms=transforms, horizon=horizon) for horizon in horizons]
