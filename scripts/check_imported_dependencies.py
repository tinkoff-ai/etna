import ast
import pathlib
from functools import lru_cache

import isort
import toml

FILE_PATH = pathlib.Path(__file__).resolve()


def lev_dist(a: str, b: str):
    """https://towardsdatascience.com/text-similarity-w-levenshtein-distance-in-python-2f7478986e75"""

    @lru_cache(None)  # for memorization
    def min_dist(s1, s2):

        if s1 == len(a) or s2 == len(b):
            return len(a) - s1 + len(b) - s2

        # no change required
        if a[s1] == b[s2]:
            return min_dist(s1 + 1, s2 + 1)

        return 1 + min(
            min_dist(s1, s2 + 1),  # insert character
            min_dist(s1 + 1, s2),  # delete character
            min_dist(s1 + 1, s2 + 1),  # replace character
        )

    return min_dist(0, 0)


def find_imported_modules(path: pathlib.Path):
    with open(path, "r") as f:
        parsed = ast.parse(f.read())

    imported_modules = set()
    for item in ast.walk(parsed):
        if isinstance(item, ast.ImportFrom):
            imported_modules.add(str(item.module).split(".")[0])
        if isinstance(item, ast.Import):
            for i in item.names:
                imported_modules.add(str(i.name).split(".")[0])
    return imported_modules


modules = set()

for path in pathlib.Path("etna").glob("**/*.py"):
    modules = modules.union(find_imported_modules(path))

modules = [i for i in modules if isort.place_module(i) == "THIRDPARTY"]

with open("pyproject.toml", "r") as f:
    pyproject = toml.load(f)

pyproject_deps = [i for i, value in pyproject["tool"]["poetry"]["dependencies"].items() if i != "python"]

missed_deps = [module for module in modules if module not in ["sklearn", "tsfresh"] and min([lev_dist(module, dep) for dep in pyproject_deps]) > 2]

if len(missed_deps) > 0:
    raise ValueError(f"Missing deps: {missed_deps}")
