import json
import pathlib

import git

CURRENT_PATH = pathlib.Path(__file__)
ROOT_PATH = CURRENT_PATH.parents[1]


def main():
    repo = git.Repo(ROOT_PATH)
    print(repo)
    print(repo.head)
    print(repo.active_branch)
    print(repo.heads)
    gh_pages = repo.heads["gh-pages"].commit
    directories = [x.path for x in gh_pages.trees]
    versions = directories

    target_dir = ROOT_PATH / "gh-pages"
    target_dir.mkdir(parents=True)
    target_file = target_dir / "versions.json"
    with target_file.open("w") as f:
        json.dump(versions, f)


if __name__ == "__main__":
    main()
