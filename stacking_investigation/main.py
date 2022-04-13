import pathlib
import subprocess

from loguru import logger

CONFIGS_PATH = pathlib.Path(__file__).parent.joinpath("configs")
RESULTS_PATH = pathlib.Path(__file__).parent.joinpath("results")


def main():
    for config_path in CONFIGS_PATH.iterdir():
        result_path = RESULTS_PATH.joinpath(config_path.name)
        if result_path in RESULTS_PATH.iterdir():
            logger.info(f"Config {config_path.name} is already done")
            continue

        logger.info(f"Started processing config: {config_path.name}")
        process = subprocess.run(["python", "run.py", "--input", config_path, "--output", result_path])
        if process.returncode != 0:
            return


if __name__ == "__main__":
    main()
