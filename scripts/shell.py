import subprocess
from typing import Optional, Tuple

import typer


class ShellError(Exception):
    ...


def shell(command: str, *args: Tuple[str], capture_output: bool = False) -> Optional[str]:
    out = subprocess.run(command.split(" ") + list(args), capture_output=capture_output)
    if out.returncode > 0:
        if capture_output:
            typer.echo(out.stdout or out.stderr)
        raise ShellError(f"Shell command returns code {out.returncode}")

    if capture_output:
        return out.stdout.decode().strip()
