"""Script for updating contributors in README.md.

Before running this script you should install `github CLI <https://github.com/cli/cli>`_.

Constant ``NUM_SKIP_FIRST_CONTRIBUTORS`` is responsible for skipping first points in contributors list.
This first positions are dedicated for former team members.

This scripts depends on the fact that contributors section goes after the team section
and license section goes after the contributors section.
"""

import json
import pathlib
import re
import subprocess
import tempfile
from typing import Any
from typing import Dict
from typing import List

ROOT_PATH = pathlib.Path(__file__).parent.resolve().parent
REPO = "/repos/tinkoff-ai/etna/contributors"
NUM_SKIP_FIRST_CONTRIBUTORS = 3


def get_contributors() -> List[Dict[str, Any]]:
    with tempfile.TemporaryFile() as fp:
        accept_format = "application/vnd.github+json"
        subprocess.run(["gh", "api", "-H", f"Accept: {accept_format}", REPO], stdout=fp)
        fp.seek(0)
        contributors = json.load(fp)
        return sorted(contributors, key=lambda x: x["contributions"], reverse=True)


def get_team_nicknames() -> List[str]:
    readme_path = ROOT_PATH.joinpath("README.md")
    with open(readme_path, "r") as fp:
        readme = fp.readlines()

    # it is expected that contributors section goes after the team section
    team_list_start = readme.index("### ETNA.Team\n")
    contributors_list_start = readme.index("### ETNA.Contributors\n")
    team_list = readme[team_list_start:contributors_list_start]
    team_list = [x.strip() for x in team_list[1:] if len(x.strip())]
    team_nicknames = [re.findall(r"https://github.com/(.*)\)", x)[0] for x in team_list]
    return team_nicknames


def write_contributors(contributors: List[Dict[str, Any]]):
    readme_path = ROOT_PATH.joinpath("README.md")
    with open(readme_path, "r") as fp:
        readme = fp.readlines()

    # it is expected that license section goes after the contributors section
    contributors_start = readme.index("### ETNA.Contributors\n")
    license_start = readme.index("## License\n")
    existing_contributors = readme[contributors_start:license_start]
    existing_contributors = [x.strip() for x in existing_contributors[1:] if len(x.strip())]
    # don't touch first NUM_SKIP_FIRST_CONTRIBUTORS
    remain_contributors = existing_contributors[:NUM_SKIP_FIRST_CONTRIBUTORS]
    remain_nicknames = [re.findall(r"https://github.com/(.*)\)", x)[0] for x in remain_contributors]
    new_contributors = [x for x in contributors if x["login"] not in remain_nicknames]

    new_lines = [f"[{x['login']}]({x['html_url']}),\n" for x in new_contributors]
    remain_lines = [f"{x}\n" for x in remain_contributors]
    contributors_lines = remain_lines + new_lines
    lines_to_write = readme[: (contributors_start + 1)] + ["\n"] + contributors_lines + ["\n"] + readme[license_start:]
    with open(readme_path, "w") as fp:
        fp.writelines(lines_to_write)


def main():
    contributors = get_contributors()
    team_nicknames = get_team_nicknames()
    external_contributors = [x for x in contributors if x["login"] not in team_nicknames]
    write_contributors(external_contributors)


if __name__ == "__main__":
    main()
