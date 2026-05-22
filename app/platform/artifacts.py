from __future__ import annotations

import gzip
import json
from pathlib import Path


def first_existing_path(*candidates: str | Path | None) -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def artifact_version(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return str(path)
    return f"{path}@{int(stat.st_mtime)}:{stat.st_size}"


def read_json_artifact(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))
