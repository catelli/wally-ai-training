import numpy as np

from wally_ai_search.utils.tiled_inference import (
    GlobalDetection,
    compute_tile_windows,
    nms_global_detections,
    shift_boxes_to_global,
)
from wally_ai_search.utils.tiled_inference import TileWindow


def test_compute_tile_windows() -> None:
    windows = compute_tile_windows(512, 512, tile_size=256, overlap=0)
    assert len(windows) == 4


def test_shift_boxes_to_global() -> None:
    window = TileWindow(x_offset=256, y_offset=0, width=256, height=256)
    boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
    shifted = shift_boxes_to_global(boxes, window)
    assert shifted[0, 0] == 266.0


def test_nms_global_detections() -> None:
    detections = [
        GlobalDetection(0, 0.9, 10, 10, 50, 50),
        GlobalDetection(0, 0.8, 12, 12, 48, 48),
        GlobalDetection(0, 0.7, 200, 200, 240, 240),
    ]
    kept = nms_global_detections(detections, iou_threshold=0.45)
    assert len(kept) == 2
