#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRINT_ONLY_ARG="${1:-}"
CONFIG_PATH="${2:-Libs/configs/config_stack.yaml}"

for PROFILE_NAME in \
  ad_screening_without_atlas_prior \
  ad_screening_without_clue_reversal \
  ad_screening_without_target_ranking \
  ad_screening_without_pkpd_prior \
  ad_screening_without_tri_gate
do
  "${SCRIPT_DIR}/submit_profile_suite.sh" "${PROFILE_NAME}" "${PRINT_ONLY_ARG}" "${CONFIG_PATH}"
done
