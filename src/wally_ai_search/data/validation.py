from pathlib import Path

from wally_ai_search.config.paths import ProjectPaths
from wally_ai_search.utils.image_utils import list_images


def validate_split(paths: ProjectPaths, split: str) -> tuple[int, int]:
    images_dir = paths.wally_images / split
    labels_dir = paths.wally_labels / split
    images = list_images(images_dir)
    label_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.is_dir() else 0
    return len(images), label_count
