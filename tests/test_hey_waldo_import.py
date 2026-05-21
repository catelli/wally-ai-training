from pathlib import Path

from wally_ai_search.data.hey_waldo_import import (
    build_scene_records,
    collect_waldo_patches,
    parse_patch_filename,
    patch_to_yolo_line,
    split_scenes,
    write_scene_labels,
)
from wally_ai_search.data.hey_waldo_import import PatchAnnotation, SceneRecord


def test_parse_patch_filename() -> None:
    assert parse_patch_filename("5_0_1") == PatchAnnotation("5", 0, 1)
    assert parse_patch_filename("bad") is None


def test_patch_to_yolo_line_normalized() -> None:
    line = patch_to_yolo_line(0, 1, 256, 2800, 1760)
    parts = line.split()
    assert parts[0] == "0"
    cx, cy, w, h = map(float, parts[1:])
    assert 0 < cx < 1 and 0 < cy < 1 and 0 < w < 1 and 0 < h < 1


def test_split_scenes_preserves_all_records() -> None:
    records = [
        SceneRecord("1", Path("1.jpg"), ()),
        SceneRecord("2", Path("2.jpg"), (PatchAnnotation("2", 0, 0),)),
    ]
    splits = split_scenes(records, val_ratio=0.5, test_ratio=0.0, seed=1)
    assert len(splits["train"]) + len(splits["val"]) == 2


def test_write_scene_labels(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "5.jpg"
    Image.new("RGB", (512, 512), color="white").save(image_path)
    record = SceneRecord("5", image_path, (PatchAnnotation("5", 0, 1),))
    label_path = tmp_path / "5.txt"
    count = write_scene_labels(record, label_path, patch_size=256)
    assert count == 1
    assert label_path.read_text(encoding="utf-8").strip().startswith("0 ")


def test_collect_waldo_patches_skips_invalid_names(tmp_path: Path) -> None:
    waldo_dir = tmp_path / "waldo"
    waldo_dir.mkdir()
    (waldo_dir / "5_0_1.jpg").touch()
    (waldo_dir / "invalid.jpg").touch()
    patches = collect_waldo_patches(waldo_dir)
    assert len(patches["5"]) == 1


def test_build_scene_records_from_fixture(tmp_path: Path) -> None:
    root = tmp_path / "Hey-Waldo"
    (root / "original-images").mkdir(parents=True)
    (root / "256" / "waldo").mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (512, 512)).save(root / "original-images" / "1.jpg")
    Image.new("RGB", (64, 64)).save(root / "256" / "waldo" / "1_0_0.jpg")

    records = build_scene_records(root, patch_size=256)
    assert len(records) == 1
    assert records[0].patches[0].row == 0
