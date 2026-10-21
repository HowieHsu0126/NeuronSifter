#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="Libs/configs/config_stack.yaml"
PRINT_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --print-only)
      PRINT_ONLY=1
      shift
      ;;
    --help|-h)
      echo "Usage: bash $0 [--config PATH] [--print-only]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
PROJECT_PREFIX="$(tr -d '[:space:]' < PROJECT_PREFIX)"
if [[ "${PROJECT_PREFIX}" != "NrnSift" ]]; then
  echo "PROJECT_PREFIX must be NrnSift." >&2
  exit 1
fi

PYTHON_BIN="$(CONFIG_PATH="${CONFIG_PATH}" python - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
print(str(config["execution_environment"]["python_executable"]))
PY
)"

COMMAND=("${PYTHON_BIN}" -u Libs/scripts/validators/validate_formal_experiment_freeze.py --no-write --config "${CONFIG_PATH}")
if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  printf 'FREEZE_REPLAY_GATE='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
sinfo
"${COMMAND[@]}"
