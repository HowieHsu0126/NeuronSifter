#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRINT_ONLY_ARG="${1:-}"
CONFIG_PATH="${2:-Libs/configs/config_stack.yaml}"
"${SCRIPT_DIR}/submit_profile_suite.sh" shared_latent_formal "${PRINT_ONLY_ARG}" "${CONFIG_PATH}"
