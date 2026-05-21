.PHONY: install install-dev lint format typecheck test import-hey-waldo train predict evaluate

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

train:
	wally-ai train

predict:
	wally-ai predict --source data/datasets/wally/images/test

evaluate:
	wally-ai evaluate
