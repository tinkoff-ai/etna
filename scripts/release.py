from enum import Enum

import click
import typer
from semver import VersionInfo as Version

from .shell import shell
from .shell import ShellError

REPO = 'https://github.com/tinkoff-ai/etna'


class Rule(str, Enum):
    prerelease = "prerelease"
    prepatch = "prepatch"
    preminor = "preminor"
    patch = "patch"
    minor = "minor"


def is_unstable(version: Version):
    return bool(version.prerelease)


def main(rule: Rule):
    try:
        shell('gh auth status')
    except ShellError:
        typer.secho(
            f'Please, auth with command:\n' +
            typer.style("gh auth login --web", bold=True)
        )
        return

    prev_version = Version.parse(
        shell('poetry version --short', capture_output=True)
    )

    package_name = shell('poetry version', capture_output=True).split(' ')[0]

    if is_unstable(prev_version) and rule in {Rule.prepatch, Rule.preminor}:
        typer.secho(
            f'\nYou should use "{Rule.prerelease}" command to update unstable releases',
            bold=True
        )
        return

    prerelease_prefix, is_prerelease = '', False
    if rule in {Rule.prerelease, Rule.prepatch, Rule.preminor}:
        prerelease_prefix, is_prerelease = 'PRE-', True

    shell(f'poetry version {rule}')
    version = shell('poetry version --short', capture_output=True)

    confirm_message = (
        '\nDo you really want to ' +
        click.style(prerelease_prefix, fg='yellow') + 'release ' +
        f'{package_name}==' + click.style(version, bold=True)
    )
    if not click.confirm(confirm_message, default=False):
        typer.echo("Ok...\n", err=True)

        shell(f'poetry version {prev_version}')
        return

    message = f':bomb: {prerelease_prefix}release {version}'

    shell(f'git checkout -b release/{version}')
    shell('git commit -am', message)
    shell(f'git push -u origin release/{version}')

    shell(f'gh pr create --title', message, '--body', f'''\
Great!
Please visit {REPO}/releases/edit/{version} to describe **release notes!**

Also you can find publishing task here {REPO}/actions/workflows/publish.yml''')

    current_branch = shell('git rev-parse --abbrev-ref HEAD', capture_output=True)
    gh_release_args = ('--prerelease', ) if is_prerelease else ()
    shell(
        f'gh release create {version}',
        '--title', message,
        '--notes', 'In progress...',
        '--target', current_branch,
        *gh_release_args,
    )
    shell('gh pr view --web')

    typer.secho('Done!', fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    typer.run(main)