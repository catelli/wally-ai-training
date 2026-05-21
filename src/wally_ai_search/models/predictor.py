from pathlib import Path
from typing import Any

from wally_ai_search.models.yolo_model import YoloModel


class YoloPredictor:
    def __init__(self, weights: str | Path) -> None:
        self.yolo = YoloModel(weights)

    def predict(self, source: str | Path, **kwargs: Any) -> Any:
        return self.yolo.predict(source=str(source), **kwargs)
