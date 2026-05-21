from pathlib import Path

from wally_ai_search.config.paths import ProjectPaths, find_project_root, get_project_paths


def test_find_project_root():
    root = find_project_root(Path(__file__).resolve().parent)
    assert (root / "pyproject.toml").exists()
    assert (root / "configs" / "dataset.yaml").exists()


def test_project_paths_layout():
    paths = get_project_paths()
    assert paths.configs.is_dir()
    assert paths.data.name == "data"
    assert paths.wally_dataset.name == "wally"
    assert paths.wally_images.name == "images"
    assert paths.runs.name == "runs"


def test_ensure_runtime_dirs(tmp_path):
    paths = ProjectPaths(root=tmp_path)
    paths.ensure_runtime_dirs()
    assert (paths.wally_images / "train").is_dir()
    assert (paths.wally_labels / "val").is_dir()
    assert paths.runs.is_dir()
