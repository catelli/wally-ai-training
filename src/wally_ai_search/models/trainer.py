from pathlib import Path
from typing import Any

from wally_ai_search.models.yolo_model import YoloModel


class YoloTrainer:
    def __init__(self, model_weights: str | Path) -> None:
        self.yolo = YoloModel(model_weights)

    def train(self, data: str | Path, **kwargs: Any) -> Any:
        return self.yolo.train(data=str(data), **kwargs)
