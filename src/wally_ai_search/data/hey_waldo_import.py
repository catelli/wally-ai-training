"""Import Hey-Waldo original scenes into YOLO detection format."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from wally_ai_search.config.paths import ProjectPaths, get_project_paths
from wally_ai_search.utils.logger import get_logger

logger = get_logger()

CLASS_ID = 0
DEFAULT_PATCH_SIZE = 256
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15


@dataclass(frozen=True)
class PatchAnnotation:
    image_id: str
    row: int
    col: int


@dataclass(frozen=True)
class SceneRecord:
    image_id: str
    source_path: Path
    patches: tuple[PatchAnnotation, ...]


def parse_patch_filename(stem: str) -> PatchAnnotation | None:
    parts = stem.split("_")
    if len(parts) != 3:
        return None
    image_id, row_str, col_str = parts
    try:
        return PatchAnnotation(image_id=image_id, row=int(row_str), col=int(col_str))
    except ValueError:
        return None


def patch_to_yolo_line(
    row: int,
    col: int,
    patch_size: int,
    image_width: int,
    image_height: int,
    class_id: int = CLASS_ID,
) -> str:
    x1 = col * patch_size
    y1 = row * patch_size
    x2 = x1 + patch_size
    y2 = y1 + patch_size
    cx = ((x1 + x2) / 2) / image_width
    cy = ((y1 + y2) / 2) / image_height
    w = patch_size / image_width
    h = patch_size / image_height
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def collect_waldo_patches(waldo_dir: Path) -> dict[str, list[PatchAnnotation]]:
    by_image: dict[str, list[PatchAnnotation]] = {}
    for path in sorted(waldo_dir.glob("*.jpg")):
        patch = parse_patch_filename(path.stem)
        if patch is None:
            logger.warning("Skipping unrecognized patch filename: {}", path.name)
            continue
        by_image.setdefault(patch.image_id, []).append(patch)
    return by_image


def build_scene_records(
    hey_waldo_root: Path,
    patch_size: int,
) -> list[SceneRecord]:
    originals_dir = hey_waldo_root / "original-images"
    waldo_dir = hey_waldo_root / str(patch_size) / "waldo"
    if not originals_dir.is_dir():
        raise FileNotFoundError(f"Missing original-images directory: {originals_dir}")
    if not waldo_dir.is_dir():
        raise FileNotFoundError(
            f"Missing waldo patches for patch size {patch_size}: {waldo_dir}"
        )

    patches_by_image = collect_waldo_patches(waldo_dir)
    records: list[SceneRecord] = []

    for source_path in sorted(originals_dir.glob("*.jpg"), key=lambda p: int(p.stem)):
        image_id = source_path.stem
        patches = tuple(patches_by_image.get(image_id, ()))
        records.append(
            SceneRecord(
                image_id=image_id,
                source_path=source_path,
                patches=patches,
            )
        )

    skipped = set(patches_by_image) - {record.image_id for record in records}
    for image_id in sorted(skipped, key=int):
        logger.warning(
            "Skipping waldo patches for image {} (no matching original scene)",
            image_id,
        )

    return records


def split_scenes(
    records: list[SceneRecord],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[SceneRecord]]:
    if not (0 <= val_ratio < 1) or not (0 <= test_ratio < 1):
        raise ValueError("val_ratio and test_ratio must be in [0, 1)")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    test_count = round(total * test_ratio)
    val_count = round(total * val_ratio)
    train_count = total - val_count - test_count

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def write_scene_labels(
    record: SceneRecord,
    label_path: Path,
    patch_size: int,
) -> int:
    with Image.open(record.source_path) as image:
        width, height = image.size

    lines = [
        patch_to_yolo_line(patch.row, patch.col, patch_size, width, height)
        for patch in record.patches
    ]
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def import_hey_waldo_dataset(
    hey_waldo_root: Path,
    paths: ProjectPaths | None = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, int]:
    """Copy Hey-Waldo originals and write YOLO labels derived from waldo patches."""
    project_paths = paths or get_project_paths()
    project_paths.ensure_runtime_dirs()
    hey_waldo_root = hey_waldo_root.resolve()

    records = build_scene_records(hey_waldo_root, patch_size)
    splits = split_scenes(records, val_ratio, test_ratio, seed)

    stats = {"scenes": len(records), "boxes": 0, "negatives": 0}

    for split_name, split_records in splits.items():
        images_dir = project_paths.wally_images / split_name
        labels_dir = project_paths.wally_labels / split_name

        for record in split_records:
            dest_image = images_dir / f"hey_waldo_{record.image_id}.jpg"
            dest_label = labels_dir / f"hey_waldo_{record.image_id}.txt"

            if dest_image.exists() and not overwrite:
                raise FileExistsError(
                    f"Dataset file already exists: {dest_image}. Use --overwrite to replace."
                )

            shutil.copy2(record.source_path, dest_image)
            box_count = write_scene_labels(record, dest_label, patch_size)
            stats["boxes"] += box_count
            if box_count == 0:
                stats["negatives"] += 1

        logger.info(
            "Split {}: {} scenes ({} with boxes)",
            split_name,
            len(split_records),
            sum(1 for r in split_records if r.patches),
        )

    archive_dir = project_paths.raw / "hey-waldo"
    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest = archive_dir / "import_manifest.txt"
    manifest.write_text(
        f"source={hey_waldo_root}\npatch_size={patch_size}\nseed={seed}\n",
        encoding="utf-8",
    )

    return {
        **stats,
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
    }
