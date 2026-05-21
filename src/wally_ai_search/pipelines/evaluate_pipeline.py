from typing import Any

from wally_ai_search.config.paths import ProjectPaths, get_project_paths
from wally_ai_search.config.settings import load_dataset_config, load_inference_config
from wally_ai_search.data.dataset import write_dataset_yaml
from wally_ai_search.models.evaluator import YoloEvaluator
from wally_ai_search.utils.logger import get_logger

logger = get_logger()


class EvaluatePipeline:
    def __init__(
        self,
        paths: ProjectPaths | None = None,
        dataset_config: dict[str, Any] | None = None,
        inference_config: dict[str, Any] | None = None,
    ) -> None:
        self.paths = paths or get_project_paths()
        self.dataset_config = dataset_config or load_dataset_config(self.paths)
        self.inference_config = inference_config or load_inference_config(self.paths)

    def run(self) -> Any:
        dataset_yaml = write_dataset_yaml(
            self.paths.runs / "wally_dataset.yaml",
            self.dataset_config,
            paths=self.paths,
        )
        weights = self.inference_config.get("weights", "yolov8n.pt")
        evaluator = YoloEvaluator(weights)
        eval_kwargs = {
            key: value
            for key, value in self.inference_config.items()
            if key not in {"weights", "source", "save", "save_txt", "save_conf", "show"}
            and value is not None
        }
        logger.info("Evaluating weights {} on {}", weights, dataset_yaml)
        return evaluator.evaluate(data=dataset_yaml, **eval_kwargs)
