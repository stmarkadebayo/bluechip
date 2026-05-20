from __future__ import annotations

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

