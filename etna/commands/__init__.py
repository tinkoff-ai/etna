from omegaconf import OmegaConf

from etna.commands.backtest_command import backtest
from etna.commands.forecast_command import forecast
from etna.commands.resolvers import mult
from etna.commands.resolvers import shift

OmegaConf.register_new_resolver("shift", shift)
OmegaConf.register_new_resolver("mult", mult)
OmegaConf.register_new_resolver("sum", lambda x, y: x + y)
