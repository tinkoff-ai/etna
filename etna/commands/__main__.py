from typing import List

from omegaconf import OmegaConf
from typer import Typer

from etna.commands import backtest
from etna.commands import forecast

app = Typer()
app.command()(forecast)
app.command()(backtest)


def shift(steps_number: int, values: List[int]) -> List[int]:
    """Shift values for steps_number steps."""
    return [v + steps_number for v in values]


OmegaConf.register_new_resolver("shift", shift)


if __name__ == "__main__":
    app()
