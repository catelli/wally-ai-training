#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -c "from wally_ai_search.config.paths import get_project_paths; get_project_paths().ensure_runtime_dirs()"
echo "Dataset and runtime directories are ready under ${ROOT}/data and ${ROOT}/runs"
