from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    markers = ("pyproject.toml", "configs", "src")
    for path in (current, *current.parents):
        if all((path / marker).exists() for marker in markers):
            return path
    return current


class ProjectPaths:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or find_project_root()).resolve()
        self.src = self.root / "src" / "wally_ai_search"
        self.configs = self.root / "configs"
        self.data = self.root / "data"
        self.raw = self.data / "raw"
        self.processed = self.data / "processed"
        self.datasets = self.data / "datasets"
        self.wally_dataset = self.datasets / "wally"
        self.wally_images = self.wally_dataset / "images"
        self.wally_labels = self.wally_dataset / "labels"
        self.runs = self.root / "runs"
        self.notebooks = self.root / "notebooks"
        self.tests = self.root / "tests"
        self.scripts = self.root / "scripts"

    def dataset_config_path(self) -> Path:
        return self.configs / "dataset.yaml"

    def training_config_path(self) -> Path:
        return self.configs / "training.yaml"

    def inference_config_path(self) -> Path:
        return self.configs / "inference.yaml"

    def ensure_runtime_dirs(self) -> None:
        for directory in (
            self.raw,
            self.processed,
            self.wally_images / "train",
            self.wally_images / "val",
            self.wally_images / "test",
            self.wally_labels / "train",
            self.wally_labels / "val",
            self.wally_labels / "test",
            self.runs,
            self.notebooks,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_project_paths() -> ProjectPaths:
    return ProjectPaths()
