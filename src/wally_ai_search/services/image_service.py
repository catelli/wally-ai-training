from __future__ import annotations

from pathlib import Path

from wally_ai_search.utils.image_utils import list_images


class ImageService:
    def list_images(self, directory: Path) -> list[Path]:
        return list_images(directory)
