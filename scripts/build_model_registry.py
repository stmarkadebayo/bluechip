from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.platform.model_registry import LocalModelRegistry  # noqa: E402
from scripts.data_utils import write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local Bluechip model/index registry.")
    parser.add_argument("--output", default="data/processed/model_registry.json")
    args = parser.parse_args()

    registry = LocalModelRegistry(registry_path=args.output)
    payload = registry.payload()
    write_json(Path(args.output), payload)


if __name__ == "__main__":
    main()
