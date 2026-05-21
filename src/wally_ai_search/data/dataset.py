from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from wally_ai_search.config.paths import ProjectPaths, get_project_paths


def build_ultralytics_dataset_yaml(
    dataset_config: dict[str, Any],
    paths: ProjectPaths | None = None,
) -> dict[str, Any]:
    root_paths = paths or get_project_paths()
    dataset_root = root_paths.root / dataset_config.get("path", "data/datasets/wally")
    names = dataset_config.get("names", {0: "wally"})
    if isinstance(names, dict):
        class_names = [names[key] for key in sorted(names, key=lambda item: int(item))]
    else:
        class_names = list(names)

    return {
        "path": str(dataset_root.resolve()),
        "train": dataset_config.get("train", "images/train"),
        "val": dataset_config.get("val", "images/val"),
        "test": dataset_config.get("test", "images/test"),
        "names": class_names,
        "nc": dataset_config.get("nc", len(class_names)),
    }


def write_dataset_yaml(
    output_path: Path,
    dataset_config: dict[str, Any],
    paths: ProjectPaths | None = None,
) -> Path:
    payload = build_ultralytics_dataset_yaml(dataset_config, paths=paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return output_path
