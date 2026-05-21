from typing import Any


def extract_detection_count(results: Any) -> int:
    if results is None:
        return 0
    total = 0
    items = results if isinstance(results, list) else [results]
    for item in items:
        boxes = getattr(item, "boxes", None)
        if boxes is not None:
            total += len(boxes)
    return total
