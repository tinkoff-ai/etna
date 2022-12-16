import json
import subprocess
from datetime import datetime

from pydantic import BaseModel, parse_obj_as


class GithubCache(BaseModel):
    id: int
    ref: str
    key: str
    version: str
    last_accessed_at: datetime
    created_at: datetime
    size_in_bytes: int


OWNER = "tinkoff-ai"
REPO = "etna"
COMMAND_GET_LIST = f'gh api -H "Accept: application/vnd.github+json" /repos/{OWNER}/{REPO}/actions/caches'
COMMAND_DELETE = (
    lambda x: f'gh api --method DELETE -H "Accept: application/vnd.github+json" /repos/{OWNER}/{REPO}/actions/caches/{x}'
)

output = subprocess.check_output(COMMAND_GET_LIST, shell=True)

cache_list = parse_obj_as(list[GithubCache], json.loads(output)["actions_caches"])


print("Total caches:", len(cache_list))

list(map(lambda x: subprocess.check_output(COMMAND_DELETE(x.id), shell=True), cache_list))
