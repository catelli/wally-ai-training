from pathlib import Path
from typing import Any

from ultralytics import YOLO


class YoloModel:
    def __init__(self, weights: str | Path) -> None:
        self.weights = str(weights)
        self._model = YOLO(self.weights)

    @property
    def model(self) -> YOLO:
        return self._model

    def train(self, **kwargs: Any) -> Any:
        return self._model.train(**kwargs)

    def predict(self, **kwargs: Any) -> Any:
        return self._model.predict(**kwargs)

    def val(self, **kwargs: Any) -> Any:
        return self._model.val(**kwargs)
