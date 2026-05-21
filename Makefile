.PHONY: install install-dev lint format typecheck test import-hey-waldo import-waldo-tiles train train-tiles predict predict-tiled evaluate

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

typecheck:
	mypy src/wally_ai_search

test:
	pytest tests -v

import-hey-waldo:
	wally-ai import-hey-waldo ~/Downloads/Hey-Waldo --overwrite

import-waldo-tiles:
	wally-ai import-waldo-tiles ~/Downloads/wally --overwrite

train:
	wally-ai train

train-tiles:
	wally-ai train --dataset-config configs/dataset_tiles.yaml --training-config configs/training_tiles.yaml

predict:
	wally-ai predict --source data/datasets/wally/images/test

predict-tiled:
	wally-ai predict-tiled --source data/datasets/wally/images/test/hey_waldo_5.jpg --inference-config configs/inference_tiled.yaml

evaluate:
	wally-ai evaluate
