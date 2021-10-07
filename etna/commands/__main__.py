from typer import Typer

from etna.commands import forecast

app = Typer()
app.command()(forecast)

if __name__ == "__main__":
    app()
