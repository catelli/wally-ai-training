from pathlib import Path
from typing import Any

from wally_ai_search.config.paths import ProjectPaths, get_project_paths
from wally_ai_search.config.settings import load_dataset_config, load_training_config
from wally_ai_search.data.dataset import write_dataset_yaml
from wally_ai_search.models.trainer import YoloTrainer
from wally_ai_search.utils.logger import get_logger

logger = get_logger()


class TrainPipeline:
    def __init__(
        self,
        paths: ProjectPaths | None = None,
        dataset_config: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
    ) -> None:
        self.paths = paths or get_project_paths()
        self.dataset_config = dataset_config or load_dataset_config(self.paths)
        self.training_config = training_config or load_training_config(self.paths)

    def run(self) -> Any:
        self.paths.ensure_runtime_dirs()
        dataset_yaml = write_dataset_yaml(
            self.paths.runs / "wally_dataset.yaml",
            self.dataset_config,
            paths=self.paths,
        )
        logger.info("Dataset manifest written to {}", dataset_yaml)

        training = dict(self.training_config)
        model_name = training.pop("model", "yolov8n.pt")
        trainer = YoloTrainer(model_name)
        train_kwargs = {key: value for key, value in training.items() if value is not None}
        logger.info("Starting training with model {}", model_name)
        return trainer.train(data=dataset_yaml, **train_kwargs)
