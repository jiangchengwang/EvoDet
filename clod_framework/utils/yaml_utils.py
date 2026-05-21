import yaml
from pathlib import Path
from typing import Any, Optional, Union


def load_yaml(path: Union[str, Path]) -> dict[str, Any]:
    if isinstance(path, str):
        path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}