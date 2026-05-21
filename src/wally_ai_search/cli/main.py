from pathlib import Path
from typing import Optional

import typer

from wally_ai_search.config.settings import load_settings
from wally_ai_search.pipelines.evaluate_pipeline import EvaluatePipeline
from wally_ai_search.pipelines.predict_pipeline import PredictPipeline
from wally_ai_search.pipelines.train_pipeline import TrainPipeline
from wally_ai_search.utils.logger import setup_logger

app = typer.Typer(
    name="wally-ai",
    help="Train, predict, and evaluate YOLO models for Wally character detection.",
    no_args_is_help=True,
)


@app.callback()
def main(
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level."),
) -> None:
    setup_logger(log_level)


@app.command("train")
def train(
    dataset_config: Optional[Path] = typer.Option(
        None, "--dataset-config", help="Path to dataset YAML config."
    ),
    training_config: Optional[Path] = typer.Option(
        None, "--training-config", help="Path to training YAML config."
    ),
) -> None:
    """Run the training pipeline."""
    from wally_ai_search.config.settings import load_dataset_config, load_training_config, load_yaml

    load_settings()
    ds_cfg = load_yaml(dataset_config) if dataset_config else load_dataset_config()
    tr_cfg = load_yaml(training_config) if training_config else load_training_config()
    pipeline = TrainPipeline(dataset_config=ds_cfg, training_config=dict(tr_cfg))
    pipeline.run()
    typer.echo("Training finished.")


@app.command("predict")
def predict(
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="Image path, directory, or glob."
    ),
    inference_config: Optional[Path] = typer.Option(
        None, "--inference-config", help="Path to inference YAML config."
    ),
) -> None:
    """Run inference on images."""
    from wally_ai_search.config.settings import load_inference_config, load_yaml

    inf_cfg = load_yaml(inference_config) if inference_config else load_inference_config()
    pipeline = PredictPipeline(inference_config=inf_cfg, source=source)
    pipeline.run()
    typer.echo("Prediction finished.")


@app.command("import-waldo-tiles")
def import_waldo_tiles(
    source: Path = typer.Argument(
        ...,
        help="Path to wally/Hey-Waldo root (must contain 256/waldo and 256/notwaldo).",
    ),
    patch_size: int = typer.Option(256, "--patch-size", help="Tile folder size (64, 128, 256)."),
    max_negative_ratio: float = typer.Option(
        3.0,
        "--max-negative-ratio",
        help="Max notwaldo tiles per waldo tile (class balance).",
    ),
    val_ratio: float = typer.Option(0.15, "--val-ratio"),
    test_ratio: float = typer.Option(0.15, "--test-ratio"),
    seed: int = typer.Option(42, "--seed"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    """Import 256px waldo/notwaldo tiles for training (Medium tiled approach)."""
    from wally_ai_search.data.waldo_tiles_import import import_waldo_tiles_dataset

    stats = import_waldo_tiles_dataset(
        waldo_root=source,
        patch_size=patch_size,
        max_negative_ratio=max_negative_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        overwrite=overwrite,
    )
    typer.echo(
        f"Imported {stats['total']} tiles "
        f"(waldo={stats['positives']}, notwaldo={stats['negatives_used']}/"
        f"{stats['negatives_available']}): "
        f"train={stats['train']} ({stats['train_waldo']} waldo), "
        f"val={stats['val']}, test={stats['test']}."
    )


@app.command("predict-tiled")
def predict_tiled(
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="Full-scene image path for tiled inference."
    ),
    inference_config: Optional[Path] = typer.Option(
        None, "--inference-config", help="Path to tiled inference YAML config."
    ),
) -> None:
    """Run tiled inference on large images (tile, detect, merge, NMS)."""
    from wally_ai_search.config.settings import load_yaml
    from wally_ai_search.pipelines.tiled_predict_pipeline import TiledPredictPipeline

    default_path = Path("configs/inference_tiled.yaml")
    inf_cfg = (
        load_yaml(inference_config)
        if inference_config
        else load_yaml(default_path)
    )
    pipeline = TiledPredictPipeline(inference_config=inf_cfg, source=source)
    detections = pipeline.run()
    typer.echo(f"Tiled prediction finished. Detections: {len(detections)}.")


@app.command("import-hey-waldo")
def import_hey_waldo(
    source: Path = typer.Argument(
        ...,
        help="Path to the Hey-Waldo dataset root (contains original-images/).",
    ),
    patch_size: int = typer.Option(
        256, "--patch-size", help="Patch size folder used for waldo/ labels (64, 128, 256)."
    ),
    val_ratio: float = typer.Option(0.15, "--val-ratio", help="Validation split ratio."),
    test_ratio: float = typer.Option(0.15, "--test-ratio", help="Test split ratio."),
    seed: int = typer.Option(42, "--seed", help="Random seed for train/val/test split."),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace existing files in data/datasets/wally/."
    ),
) -> None:
    """Import Hey-Waldo original scenes with YOLO boxes from waldo patches (option B)."""
    from wally_ai_search.data.hey_waldo_import import import_hey_waldo_dataset

    stats = import_hey_waldo_dataset(
        hey_waldo_root=source,
        patch_size=patch_size,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        overwrite=overwrite,
    )
    typer.echo(
        f"Imported {stats['scenes']} scenes: "
        f"train={stats['train']}, val={stats['val']}, test={stats['test']}, "
        f"boxes={stats['boxes']}, negatives={stats['negatives']}."
    )


@app.command("evaluate")
def evaluate(
    dataset_config: Optional[Path] = typer.Option(
        None, "--dataset-config", help="Path to dataset YAML config."
    ),
    inference_config: Optional[Path] = typer.Option(
        None, "--inference-config", help="Path to inference YAML config."
    ),
) -> None:
    """Evaluate model metrics on the validation split."""
    from wally_ai_search.config.settings import (
        load_dataset_config,
        load_inference_config,
        load_yaml,
    )

    ds_cfg = load_yaml(dataset_config) if dataset_config else load_dataset_config()
    inf_cfg = load_yaml(inference_config) if inference_config else load_inference_config()
    pipeline = EvaluatePipeline(dataset_config=ds_cfg, inference_config=inf_cfg)
    pipeline.run()
    typer.echo("Evaluation finished.")


if __name__ == "__main__":
    app()
