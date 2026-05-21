from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from wally_ai_search.config.paths import ProjectPaths, get_project_paths


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WALLY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=lambda: get_project_paths().root)
    dataset_config: Path = Field(
        default_factory=lambda: get_project_paths().dataset_config_path()
    )
    training_config: Path = Field(
        default_factory=lambda: get_project_paths().training_config_path()
    )
    inference_config: Path = Field(
        default_factory=lambda: get_project_paths().inference_config_path()
    )
    log_level: str = "INFO"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_settings() -> AppSettings:
    return AppSettings()


def load_dataset_config(paths: ProjectPaths | None = None) -> dict[str, Any]:
    root_paths = paths or get_project_paths()
    return load_yaml(root_paths.dataset_config_path())


def load_training_config(paths: ProjectPaths | None = None) -> dict[str, Any]:
    root_paths = paths or get_project_paths()
    return load_yaml(root_paths.training_config_path())


def load_inference_config(paths: ProjectPaths | None = None) -> dict[str, Any]:
    root_paths = paths or get_project_paths()
    return load_yaml(root_paths.inference_config_path())
