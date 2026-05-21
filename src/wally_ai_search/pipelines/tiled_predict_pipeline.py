from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from wally_ai_search.config.paths import ProjectPaths, get_project_paths
from wally_ai_search.config.settings import load_inference_config
from wally_ai_search.models.predictor import YoloPredictor
from wally_ai_search.utils.logger import get_logger
from wally_ai_search.utils.tiled_inference import (
    GlobalDetection,
    compute_tile_windows,
    extract_tile,
    nms_global_detections,
    save_annotated_image,
    shift_boxes_to_global,
)

logger = get_logger()


class TiledPredictPipeline:
    def __init__(
        self,
        paths: ProjectPaths | None = None,
        inference_config: dict[str, Any] | None = None,
        source: str | Path | None = None,
    ) -> None:
        self.paths = paths or get_project_paths()
        self.inference_config = dict(inference_config or load_inference_config(self.paths))
        if source is not None:
            self.inference_config["source"] = str(source)

    def run(self) -> list[GlobalDetection]:
        weights = self.inference_config.pop("weights", "runs/wally_train/weights/best.pt")
        source = Path(self.inference_config.pop("source", self.paths.wally_images / "test"))
        tile_size = int(self.inference_config.pop("tile_size", 256))
        overlap = int(self.inference_config.pop("tile_overlap", 64))
        conf = float(self.inference_config.pop("conf", 0.25))
        iou = float(self.inference_config.pop("iou", 0.45))
        merge_iou = float(self.inference_config.pop("merge_iou", 0.45))
        output_dir = Path(
            self.inference_config.pop(
                "project",
                self.paths.runs,
            )
        )
        run_name = self.inference_config.pop("name", "wally_tiled_predict")
        save = bool(self.inference_config.pop("save", True))

        image = Image.open(source).convert("RGB")
        windows = compute_tile_windows(image.width, image.height, tile_size, overlap)
        predictor = YoloPredictor(weights)
        merged: list[GlobalDetection] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for index, window in enumerate(windows):
                tile = extract_tile(image, window, tile_size)
                tile_path = temp_path / f"tile_{index}.jpg"
                tile.save(tile_path)

                results = predictor.predict(
                    source=str(tile_path),
                    imgsz=tile_size,
                    conf=conf,
                    iou=iou,
                    save=False,
                    verbose=False,
                    **{
                        key: value
                        for key, value in self.inference_config.items()
                        if value is not None
                    },
                )

                result = results[0]
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                global_boxes = shift_boxes_to_global(boxes, window)

                for class_id, score, box in zip(classes, confidences, global_boxes):
                    merged.append(
                        GlobalDetection(
                            class_id=int(class_id),
                            confidence=float(score),
                            x1=float(box[0]),
                            y1=float(box[1]),
                            x2=float(box[2]),
                            y2=float(box[3]),
                        )
                    )

        detections = nms_global_detections(merged, iou_threshold=merge_iou)
        logger.info("Tiled prediction on {}: {} detections after NMS", source, len(detections))

        if save:
            out_dir = output_dir / run_name
            save_annotated_image(image, detections, out_dir / f"{source.stem}_tiled.jpg")

        return detections
