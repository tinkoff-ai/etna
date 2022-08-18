import json
from hashlib import md5
from typing import Callable
from typing import Any
import random
import time


def config_hash(config: dict):
    """Compute hash of given ``config``."""
    return md5(json.dumps(config, sort_keys=True).encode()).hexdigest()


def retry(func: Callable[..., Any], max_retries: int = 5, sleep_time: int = 10, jitter: int = 10) -> Any:
    rng = random.SystemRandom()
    for i in range(max_retries):
        try:
            value = func()
            return value
        except Exception as e:
            if i < max_retries:
                time.sleep(rng.random() * jitter + sleep_time)
                continue
            else:
                raise e
