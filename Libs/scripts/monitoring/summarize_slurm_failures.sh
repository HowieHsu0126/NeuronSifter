#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="Output/logs"
PRINT_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --print-only)
      PRINT_ONLY=1
      shift
      ;;
    --help|-h)
      echo "Usage: bash $0 [--log-root PATH] [--print-only]"
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
COMMAND=(rg -n "Traceback|CUDA out of memory|OutOfMemory|non-finite|DependencyNeverSatisfied|FAILED|CANCELLED|TIMEOUT|Node failure|No space left|I/O error" "${LOG_ROOT}")
if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  printf 'SUMMARIZE_FAILURES='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
"${COMMAND[@]}"
