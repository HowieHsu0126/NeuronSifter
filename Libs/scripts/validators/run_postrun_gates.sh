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

COMMANDS=(
  "${PYTHON_BIN} -u Libs/exp/evaluation/evaluate_published_ad_sift_reproduction.py --config ${CONFIG_PATH} --allow-pending"
  "${PYTHON_BIN} -u Libs/exp/aggregation/build_paper_grade_tables.py --config ${CONFIG_PATH} --dry-run"
  "${PYTHON_BIN} -u Libs/exp/aggregation/build_completion_audit.py --config ${CONFIG_PATH} --dry-run"
  "${PYTHON_BIN} -u Libs/scripts/validators/validate_paper_grade_full_suite.py --config ${CONFIG_PATH} --mode postrun"
)

if [[ "${PRINT_ONLY}" -eq 0 ]]; then
  sinfo
fi

for command in "${COMMANDS[@]}"; do
  if [[ "${PRINT_ONLY}" -eq 1 ]]; then
    printf 'POSTRUN_GATE=%q\n' "${command}"
    continue
  fi
  read -r -a command_parts <<< "${command}"
  "${command_parts[@]}"
done
