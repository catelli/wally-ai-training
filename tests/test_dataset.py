from wally_ai_search.config.paths import ProjectPaths
from wally_ai_search.data.dataset import build_ultralytics_dataset_yaml, write_dataset_yaml


def test_build_ultralytics_dataset_yaml(tmp_path):
    paths = ProjectPaths(root=tmp_path)
    config = {
        "path": "data/datasets/wally",
        "train": "images/train",
        "val": "images/val",
        "names": {0: "wally"},
        "nc": 1,
    }
    payload = build_ultralytics_dataset_yaml(config, paths=paths)
    assert payload["names"] == ["wally"]
    assert payload["nc"] == 1
    assert "train" in payload


def test_write_dataset_yaml(tmp_path):
    paths = ProjectPaths(root=tmp_path)
    output = paths.runs / "test_dataset.yaml"
    write_dataset_yaml(output, {"path": "data/datasets/wally", "names": {0: "wally"}, "nc": 1}, paths=paths)
    assert output.exists()
    assert "wally" in output.read_text()
