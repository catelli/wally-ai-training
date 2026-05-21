#!/usr/bin/env python3
"""Run Hey-Waldo import (requires Python 3.11+ and pip install -e .)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wally_ai_search.data.hey_waldo_import import import_hey_waldo_dataset  # noqa: E402


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Import Hey-Waldo into YOLO layout.")
    parser.add_argument("source", type=Path, help="Hey-Waldo dataset root")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    stats = import_hey_waldo_dataset(
        hey_waldo_root=args.source,
        patch_size=args.patch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(stats)


if __name__ == "__main__":
    main()
