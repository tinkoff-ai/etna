from enum import Enum
from typing import List

from hydra_slayer import get_from_params

from etna.auto.pool.templates import DEFAULT
from etna.auto.pool.utils import fill_template
from etna.pipeline import Pipeline


class PoolGenerator:
    """Generate a pool of pipelines from given config templates in hydra format."""

    def __init__(self, configs_template: List[dict]):
        """
        Initialize with a list of config templates in hydra format.

        Parameters
        ----------
        configs_template:
            list of template configs in hydra format

        Notes
        -----
        Hydra configs templates:
        ::
            {
                '_target_': 'etna.pipeline.Pipeline',
                'horizon': '${__aux__.horizon}',
                'model': {'_target_': 'etna.models.ProphetModel'}
            }
        Values to be interpolated should be in the form of ``${__aux__.key}``
        """
        self.configs_template = configs_template

    def generate(self, horizon: int) -> List[Pipeline]:
        """
        Fill templates with args.

        Parameters
        ----------
        horizon:
            horizon to forecast
        """
        filled_templates: List[dict] = [fill_template(config, {"horizon": horizon}) for config in self.configs_template]
        return [get_from_params(**filled_template) for filled_template in filled_templates]


class Pool(Enum):
    """Predefined pools of pipelines."""

    default = PoolGenerator(configs_template=DEFAULT)  # type: ignore
