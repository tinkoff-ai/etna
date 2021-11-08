from pathlib import Path

import typer

from .shell import shell

CURRENT_PATH = Path(__file__)
ROOT_PATH = CURRENT_PATH.parents[1]
NOTEBOOKS_FOLDER = ROOT_PATH / "examples"

NOTEBOOKS_TO_SKIP = []


def run_notebooks():
    for notebook_path in NOTEBOOKS_FOLDER.glob("*.ipynb"):
        if notebook_path.name in NOTEBOOKS_TO_SKIP:
            typer.echo(f"Skipping {notebook_path}")
            continue
        typer.echo(f"Running {notebook_path}")
        shell(
            f"poetry run python -m jupyter nbconvert --ExecutePreprocessor.kernel_name=python3 --to notebook --execute {notebook_path}"
        )


if __name__ == "__main__":
    run_notebooks()
