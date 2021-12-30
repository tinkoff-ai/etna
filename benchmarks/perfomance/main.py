import json
import pathlib
import subprocess
import threading

import hydra
from omegaconf import DictConfig
from etna.commands import *


from scripts.utils import get_perfomance_dataframe_py_spy

FILE_FOLDER = pathlib.Path(__file__).parent.resolve()


def output_reader(proc):
    for line in iter(proc.stdout.readline, b""):
        print(line.decode("utf-8"), end="")


@hydra.main(config_path="configs/", config_name="config")
def bench(cfg: DictConfig) -> None:
    proc = subprocess.Popen(
        ["py-spy", "record", "-o", "speedscope.json", "-f", "speedscope", "python", FILE_FOLDER / "scripts" / "run.py"],
        stdout=subprocess.PIPE,
        #stderr=subprocess.PIPE,
    )
    t = threading.Thread(target=output_reader, args=(proc,))
    t.start()
    proc.wait()

    #if proc.returncode != 0:
    #    for i in proc.stderr.readlines():
    #        print(i.decode())

    with open("speedscope.json", "r") as f:
        py_spy_dict = json.load(f)

    df = get_perfomance_dataframe_py_spy(py_spy_dict, top=cfg.top, pattern_to_filter=cfg.pattern_to_filter)
    df["line"] = df["line"].apply(lambda x: str(x).strip().replace("\\n", ""))
    df.to_csv("py_spy.csv")


if __name__ == "__main__":
    bench()
