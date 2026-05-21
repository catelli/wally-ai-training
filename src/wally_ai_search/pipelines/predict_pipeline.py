from __future__ import annotations

from typing import Any

from wally_ai_search.config.paths import ProjectPaths, get_project_paths
from wally_ai_search.config.settings import load_inference_config
from wally_ai_search.models.predictor import YoloPredictor
from wally_ai_search.utils.logger import get_logger

logger = get_logger()


class PredictPipeline:
    def __init__(
        self,
        paths: ProjectPaths | None = None,
        inference_config: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> None:
        self.paths = paths or get_project_paths()
        self.inference_config = dict(inference_config or load_inference_config(self.paths))
        if source is not None:
            self.inference_config["source"] = source

    def run(self) -> Any:
        weights = self.inference_config.pop("weights", "yolov8n.pt")
        source = self.inference_config.pop("source", str(self.paths.wally_images / "test"))
        predictor = YoloPredictor(weights)
        predict_kwargs = {
            key: value
            for key, value in self.inference_config.items()
            if value is not None
        }
        logger.info("Running prediction on {} with weights {}", source, weights)
        return predictor.predict(source=source, **predict_kwargs)
