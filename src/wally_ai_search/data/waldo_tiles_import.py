"""Import Hey-Waldo / wally tile patches for small-object detection training."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from wally_ai_search.config.paths import ProjectPaths, get_project_paths
from wally_ai_search.utils.logger import get_logger

logger = get_logger()

FULL_TILE_LABEL = "0 0.500000 0.500000 1.000000 1.000000\n"
WALDO_PREFIX = "waldo_tile_"
NEGATIVE_PREFIX = "notwaldo_tile_"


@dataclass(frozen=True)
class TileSample:
    source_path: Path
    is_waldo: bool
    stem: str


def collect_tile_samples(waldo_root: Path, patch_size: int) -> tuple[list[TileSample], list[TileSample]]:
    waldo_dir = waldo_root / str(patch_size) / "waldo"
    notwaldo_dir = waldo_root / str(patch_size) / "notwaldo"
    if not waldo_dir.is_dir() or not notwaldo_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {waldo_dir} and {notwaldo_dir} (Medium-style 256/waldo layout)."
        )

    positives = [
        TileSample(path, True, path.stem)
        for path in sorted(waldo_dir.glob("*.jpg"))
    ]
    negatives = [
        TileSample(path, False, path.stem)
        for path in sorted(notwaldo_dir.glob("*.jpg"))
    ]
    return positives, negatives


def sample_negatives(
    negatives: list[TileSample],
    positive_count: int,
    max_negative_ratio: float,
    seed: int,
) -> list[TileSample]:
    if max_negative_ratio <= 0:
        return []
    cap = max(positive_count, int(positive_count * max_negative_ratio))
    cap = min(cap, len(negatives))
    return random.Random(seed).sample(negatives, cap)


def split_samples(
    samples: list[TileSample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[TileSample]]:
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1")

    by_class: dict[bool, list[TileSample]] = {True: [], False: []}
    for sample in samples:
        by_class[sample.is_waldo].append(sample)

    splits: dict[str, list[TileSample]] = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)

    for class_samples in by_class.values():
        shuffled = list(class_samples)
        rng.shuffle(shuffled)
        total = len(shuffled)
        test_count = round(total * test_ratio)
        val_count = round(total * val_ratio)
        train_count = total - val_count - test_count
        splits["train"].extend(shuffled[:train_count])
        splits["val"].extend(shuffled[train_count : train_count + val_count])
        splits["test"].extend(shuffled[train_count + val_count :])

    for split_name in splits:
        rng.shuffle(splits[split_name])

    return splits


def copy_tile_sample(
    sample: TileSample,
    images_dir: Path,
    labels_dir: Path,
    overwrite: bool,
) -> None:
    prefix = WALDO_PREFIX if sample.is_waldo else NEGATIVE_PREFIX
    dest_name = f"{prefix}{sample.stem}.jpg"
    dest_image = images_dir / dest_name
    dest_label = labels_dir / f"{prefix}{sample.stem}.txt"

    if dest_image.exists() and not overwrite:
        raise FileExistsError(f"Dataset file already exists: {dest_image}")

    shutil.copy2(sample.source_path, dest_image)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if sample.is_waldo:
        dest_label.write_text(FULL_TILE_LABEL, encoding="utf-8")
    else:
        dest_label.write_text("", encoding="utf-8")


def import_waldo_tiles_dataset(
    waldo_root: Path,
    paths: ProjectPaths | None = None,
    patch_size: int = 256,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_negative_ratio: float = 3.0,
    seed: int = 42,
    overwrite: bool = False,
    dataset_name: str = "wally_tiles",
) -> dict[str, int]:
    """
    Build a YOLO tile dataset (Medium article approach).

    Positive tiles (waldo/) get a full-frame box. Negative tiles (notwaldo/) are
    subsampled to reduce class imbalance.
    """
    project_paths = paths or get_project_paths()
    waldo_root = waldo_root.resolve()

    positives, negatives = collect_tile_samples(waldo_root, patch_size)
    selected_negatives = sample_negatives(negatives, len(positives), max_negative_ratio, seed)
    all_samples = positives + selected_negatives
    splits = split_samples(all_samples, val_ratio, test_ratio, seed)

    dataset_root = project_paths.datasets / dataset_name
    stats: dict[str, int] = {
        "positives": len(positives),
        "negatives_available": len(negatives),
        "negatives_used": len(selected_negatives),
        "total": len(all_samples),
    }

    for split_name, split_records in splits.items():
        images_dir = dataset_root / "images" / split_name
        labels_dir = dataset_root / "labels" / split_name
        images_dir.mkdir(parents=True, exist_ok=True)

        for sample in split_records:
            copy_tile_sample(sample, images_dir, labels_dir, overwrite)

        waldo_count = sum(1 for s in split_records if s.is_waldo)
        stats[split_name] = len(split_records)
        stats[f"{split_name}_waldo"] = waldo_count
        logger.info(
            "Split {}: {} tiles ({} waldo, {} notwaldo)",
            split_name,
            len(split_records),
            waldo_count,
            len(split_records) - waldo_count,
        )

    manifest_dir = project_paths.raw / dataset_name
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "import_manifest.txt").write_text(
        f"source={waldo_root}\npatch_size={patch_size}\nmax_negative_ratio={max_negative_ratio}\n",
        encoding="utf-8",
    )

    return stats
