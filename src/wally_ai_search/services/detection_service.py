from __future__ import annotations

from pathlib import Path
from typing import Any

from wally_ai_search.models.predictor import YoloPredictor
from wally_ai_search.utils.metrics import extract_detection_count


class DetectionService:
    def __init__(self, weights: str | Path) -> None:
        self.predictor = YoloPredictor(weights)

    def detect(self, source: str | Path, **kwargs: Any) -> tuple[Any, int]:
        results = self.predictor.predict(source=source, **kwargs)
        return results, extract_detection_count(results)
