# Wally AI Search

Train, evaluate, and run YOLO models to detect the Wally character in images.

## Requirements

- Python 3.11+
- CUDA-capable GPU (recommended for training)

## Setup

```bash
cd wally-ai-training
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Dataset layout

Place your YOLO-format dataset under `data/datasets/wally/`:

```txt
data/datasets/wally/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Each image in `images/<split>/` needs a matching `.txt` label file in `labels/<split>/` (same basename, YOLO normalized bbox format).

Paths and class names are configured in `configs/dataset.yaml`.

## Import Hey-Waldo (option B — full-scene detection)

The [Hey-Waldo](https://github.com/brendenlake/Hey-Waldo) dataset provides `original-images/` (full book scenes) and `256/waldo/` patches whose filenames encode grid cells (`{imageId}_{row}_{col}.jpg`). The import command copies originals into the YOLO layout and writes bounding boxes from those cells.

```bash
wally-ai import-hey-waldo ~/Downloads/Hey-Waldo --overwrite
# or
make import-hey-waldo
```

This produces 19 scenes split into train/val/test (default 70% / 15% / 15%, seed 42). Scenes without a Waldo patch (e.g. image 8, 15) are kept as negatives with empty label files. Patch `21_*` is skipped because there is no matching original.

Then train as usual:

```bash
wally-ai train
```

For manual refinement, edit labels under `data/datasets/wally/labels/` or re-annotate in LabelImg / Roboflow and re-run import with `--overwrite`.

## Tiled training (recommended — [Medium article](https://medium.com/swlh/find-waldo-with-yolov2-809db787bbdf))

Waldo is tiny in full-page images; training at `imgsz: 640` on 19 scenes often fails to learn. The article trains on **256×256 tiles** where Waldo fills the tile, then runs **tiled inference** on full scenes.

Import tiles from `~/Downloads/wally` (or Hey-Waldo — same layout: `256/waldo`, `256/notwaldo`):

```bash
wally-ai import-waldo-tiles ~/Downloads/wally --overwrite
make train-tiles
```

Train uses `configs/dataset_tiles.yaml` and `configs/training_tiles.yaml` (`imgsz: 256`, 150 epochs).

Detect Waldo on a full book page:

```bash
wally-ai predict-tiled -s data/datasets/wally/images/test/hey_waldo_5.jpg \
  --inference-config configs/inference_tiled.yaml
```

Output: `runs/wally_tiled_predict/<image>_tiled.jpg`

## Train

```bash
wally-ai train
# or
make train
```

Override config files:

```bash
wally-ai train --training-config configs/training.yaml --dataset-config configs/dataset.yaml
```

Training outputs are written under `runs/`.

## Predict

```bash
wally-ai predict --source path/to/images
```

## Evaluate

```bash
wally-ai evaluate
```

## Development

```bash
make lint
make test
make typecheck
```

## Project layout

See `.cursor/rules/wally-ai-search.mdc` for architecture and conventions.
