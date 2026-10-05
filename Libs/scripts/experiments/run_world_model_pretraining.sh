#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_entrypoint_lib.sh"
entrypoint_run_task "world_model_pretraining" "$@"
