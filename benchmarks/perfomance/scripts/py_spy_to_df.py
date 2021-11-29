import argparse
import json
import pathlib
from typing import Optional

from scripts.utils import get_perfomance_dataframe_py_spy

args = argparse.ArgumentParser()
args.add_argument("--path", type=str, required=True)
args.add_argument("--top", type=int, default=None)
args.add_argument("--pattern-to-filter", type=str, required=True)


def py_spy_to_csv(path: str, py_spy_dict: dict, top: Optional[int], pattern_to_filter: str):
    df = get_perfomance_dataframe_py_spy(py_spy_dict, top=top, pattern_to_filter=pattern_to_filter)
    df["line"] = df["line"].apply(lambda x: str(x).strip().replace("\\n", ""))
    df.to_csv(path)


if __name__ == "__main__":
    args = args.parse_args()
    path = pathlib.Path(args.path).resolve()
    with open(path / "speedscope.json", "r") as f:
        py_spy_dict = json.load(f)
    py_spy_to_csv(path / "py_spy.csv", py_spy_dict=py_spy_dict, top=args.top, pattern_to_filter=args.pattern_to_filter)
