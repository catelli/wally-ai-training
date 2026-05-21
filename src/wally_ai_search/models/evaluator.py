from __future__ import annotations

from pathlib import Path
from typing import Any

from wally_ai_search.models.yolo_model import YoloModel


class YoloEvaluator:
    def __init__(self, weights: str | Path) -> None:
        self.yolo = YoloModel(weights)

    def evaluate(self, data: str | Path, **kwargs: Any) -> Any:
        return self.yolo.val(data=str(data), **kwargs)
