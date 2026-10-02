#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_entrypoint_lib.sh"

entrypoint_parse_common "$@"
REPO_ROOT="$(entrypoint_repo_root)"
cd "${REPO_ROOT}"
entrypoint_check_project_prefix "${REPO_ROOT}"

DEVELOPER_ENTRYPOINT="$(CONFIG_PATH="${CONFIG_PATH}" python - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
print(str(config["paper_grade_full_suite"]["developer_entrypoint"]))
PY
)"

DEVELOPER_ARGS=("--config" "${CONFIG_PATH}")
if [[ "${PRINT_ONLY_VALUE}" -eq 1 || "${DRY_RUN_VALUE}" -eq 1 || "${NO_SUBMIT_VALUE}" -eq 1 ]]; then
  DEVELOPER_ARGS+=("--print-only")
fi

printf 'RESOLVED_CONFIG_PATH=%s\n' "${CONFIG_PATH}"
printf 'SUITE_CONFIG_PATH=%s\n' "${SUITE_CONFIG_PATH}"
printf 'TASK=%s\n' "paper_grade_full_suite"
printf 'MODEL_FAMILY=%s\n' "${MODEL_FAMILY_VALUE}"
printf 'EXPERIMENT_FAMILY=%s\n' "${EXPERIMENT_FAMILY_VALUE}"
printf 'SEED=%s\n' "${SEED_VALUE}"
printf 'SLURM_JOB_NAME=%s\n' "${PROJECT_PREFIX}_paper_grade_full_suite"
printf 'LOG_PATH=%s\n' "Output/logs/print_only/paper_grade_full_suite"
printf 'SBATCH_COMMAND='
printf '%q ' bash "${DEVELOPER_ENTRYPOINT}" "${DEVELOPER_ARGS[@]}"
printf '\n'

if [[ "${PRINT_ONLY_VALUE}" -eq 0 && "${DRY_RUN_VALUE}" -eq 0 && "${NO_SUBMIT_VALUE}" -eq 0 ]]; then
  bash Libs/scripts/validators/run_prelaunch_gates.sh --config "${CONFIG_PATH}"
fi

bash "${DEVELOPER_ENTRYPOINT}" "${DEVELOPER_ARGS[@]}"
