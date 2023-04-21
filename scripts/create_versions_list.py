import json
import pathlib
import tempfile

import git

DOCUMENTATION_URL = "https://github.com/tinkoff-ai/etna-docs.git"
CURRENT_PATH = pathlib.Path(__file__)
ROOT_PATH = CURRENT_PATH.parents[1]


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = git.Repo.clone_from(DOCUMENTATION_URL, temp_dir)
        gh_pages = repo.heads["main"].commit.tree
        directories = [x.path for x in gh_pages.trees]
        versions = directories

    target_dir = ROOT_PATH / "gh-pages"
    target_dir.mkdir(parents=True)
    target_file = target_dir / "versions.json"
    with target_file.open("w") as f:
        json.dump(versions, f)


if __name__ == "__main__":
    main()
