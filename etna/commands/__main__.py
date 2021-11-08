from typer import Typer

from etna.commands import backtest
from etna.commands import forecast

app = Typer()
app.command()(forecast)
app.command()(backtest)


if __name__ == "__main__":
    app()
