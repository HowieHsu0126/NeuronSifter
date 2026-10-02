#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="Output/audits/slurm_log_collection"
PRINT_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --print-only)
      PRINT_ONLY=1
      shift
      ;;
    --help|-h)
      echo "Usage: bash $0 [--output-dir PATH] [--print-only]"
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
COMMAND=(rsync -a --include "${PROJECT_PREFIX}_*.out" --include "${PROJECT_PREFIX}_*.err" --include "${PROJECT_PREFIX}_*.sbatch" --exclude "*" Output/logs/ "${OUTPUT_DIR}/")
if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  printf 'COLLECT_LOGS='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
mkdir -p "${OUTPUT_DIR}"
"${COMMAND[@]}"
