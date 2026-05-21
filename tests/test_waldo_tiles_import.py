from pathlib import Path

from wally_ai_search.data.waldo_tiles_import import (
    collect_tile_samples,
    sample_negatives,
    split_samples,
)
from wally_ai_search.data.waldo_tiles_import import TileSample


def test_sample_negatives_caps_count() -> None:
    negatives = [TileSample(Path(f"n{i}.jpg"), False, f"n{i}") for i in range(10)]
    selected = sample_negatives(negatives, positive_count=3, max_negative_ratio=2.0, seed=1)
    assert len(selected) == 6


def test_split_samples_stratified() -> None:
    samples = [
        TileSample(Path("w.jpg"), True, "w"),
        TileSample(Path("n.jpg"), False, "n"),
    ]
    splits = split_samples(samples, val_ratio=0.5, test_ratio=0.0, seed=0)
    assert len(splits["train"]) + len(splits["val"]) == 2


def test_collect_tile_samples_from_fixture(tmp_path: Path) -> None:
    root = tmp_path / "wally"
    (root / "256" / "waldo").mkdir(parents=True)
    (root / "256" / "notwaldo").mkdir(parents=True)
    (root / "256" / "waldo" / "1_0_0.jpg").touch()
    (root / "256" / "notwaldo" / "1_0_1.jpg").touch()
    positives, negatives = collect_tile_samples(root, 256)
    assert len(positives) == 1
    assert len(negatives) == 1
