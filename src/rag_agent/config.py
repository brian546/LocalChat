from functools import lru_cache
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "agent.yaml"
PROJECT_ROOT = CONFIG_PATH.parent.parent


@lru_cache(maxsize=1)
def load_agent_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path