#!/usr/bin/env bash
set -euo pipefail

entrypoint_repo_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd
}

entrypoint_usage() {
  local script_name="$1"
  cat <<USAGE
Usage: bash ${script_name} [--config PATH] [--suite-config PATH] [--seed N] [--task NAME] [--model-family NAME] [--experiment-family NAME] [--partition NAME] [--nodelist NODE] [--time TIME] [--mem MEM] [--gres GRES] [--job-name-prefix PREFIX] [--dependency afterok:JOBID] [--output-dir PATH] [--freeze-manifest PATH] [--run-id ID] [--checkpoint PATH] [--dry-run] [--print-only] [--no-submit] [--extra-args "ARGS"]
USAGE
}

entrypoint_parse_common() {
  CONFIG_PATH="Libs/configs/config_stack.yaml"
  SUITE_CONFIG_PATH="Libs/configs/exp/paper_grade_full_suite.yaml"
  SEED_VALUE=""
  TASK_VALUE=""
  MODEL_FAMILY_VALUE="neuronsifter_mainline"
  EXPERIMENT_FAMILY_VALUE=""
  PARTITION_VALUE=""
  NODELIST_VALUE=""
  TIME_VALUE=""
  MEM_VALUE=""
  GRES_VALUE=""
  JOB_NAME_PREFIX_VALUE=""
  DEPENDENCY_VALUE=""
  OUTPUT_DIR_VALUE=""
  FREEZE_MANIFEST_VALUE="Output/audits/formal_experiment_freeze_manifest.json"
  RUN_ID_VALUE=""
  CHECKPOINT_VALUE=""
  DRY_RUN_VALUE=0
  PRINT_ONLY_VALUE=0
  NO_SUBMIT_VALUE=0
  DEPENDENCY_JOB_ID_VALUE=""
  EXTRA_ARGS_VALUE=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --config)
        CONFIG_PATH="$2"
        shift 2
        ;;
      --suite-config)
        SUITE_CONFIG_PATH="$2"
        shift 2
        ;;
      --seed)
        SEED_VALUE="$2"
        shift 2
        ;;
      --task)
        TASK_VALUE="$2"
        shift 2
        ;;
      --model-family)
        MODEL_FAMILY_VALUE="$2"
        shift 2
        ;;
      --experiment-family)
        EXPERIMENT_FAMILY_VALUE="$2"
        shift 2
        ;;
      --partition)
        PARTITION_VALUE="$2"
        shift 2
        ;;
      --nodelist)
        NODELIST_VALUE="$2"
        shift 2
        ;;
      --time)
        TIME_VALUE="$2"
        shift 2
        ;;
      --mem)
        MEM_VALUE="$2"
        shift 2
        ;;
      --gres)
        GRES_VALUE="$2"
        shift 2
        ;;
      --job-name-prefix)
        JOB_NAME_PREFIX_VALUE="$2"
        shift 2
        ;;
      --dependency)
        DEPENDENCY_VALUE="$2"
        shift 2
        ;;
      --output-dir)
        OUTPUT_DIR_VALUE="$2"
        shift 2
        ;;
      --freeze-manifest)
        FREEZE_MANIFEST_VALUE="$2"
        shift 2
        ;;
      --run-id)
        RUN_ID_VALUE="$2"
        shift 2
        ;;
      --checkpoint)
        CHECKPOINT_VALUE="$2"
        shift 2
        ;;
      --dependency-job-id)
        DEPENDENCY_JOB_ID_VALUE="$2"
        shift 2
        ;;
      --dry-run)
        DRY_RUN_VALUE=1
        shift
        ;;
      --print-only)
        PRINT_ONLY_VALUE=1
        shift
        ;;
      --no-submit)
        NO_SUBMIT_VALUE=1
        shift
        ;;
      --extra-args)
        EXTRA_ARGS_VALUE="$2"
        shift 2
        ;;
      --help|-h)
        entrypoint_usage "$0"
        exit 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        entrypoint_usage "$0" >&2
        exit 2
        ;;
    esac
  done
}

entrypoint_check_project_prefix() {
  local repo_root="$1"
  PROJECT_PREFIX="$(tr -d '[:space:]' < "${repo_root}/PROJECT_PREFIX")"
  if [[ "${PROJECT_PREFIX}" != "NrnSift" ]]; then
    echo "PROJECT_PREFIX must be NrnSift for NeuronSifter paper-grade entrypoints." >&2
    exit 1
  fi
}

entrypoint_python_bin() {
  local config_path="$1"
  CONFIG_PATH="${config_path}" python - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
print(str(config["execution_environment"]["python_executable"]))
PY
}

entrypoint_registry_value() {
  local config_path="$1"
  local task_name="$2"
  local field_name="$3"
  CONFIG_PATH="${config_path}" TASK_NAME="${task_name}" FIELD_NAME="${field_name}" python - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
entry = config["paper_grade_full_suite"]["entrypoint_registry"][os.environ["TASK_NAME"]]
value = entry.get(os.environ["FIELD_NAME"], "")
print("" if value is None else str(value))
PY
}

entrypoint_resource_values() {
  local config_path="$1"
  local task_name="$2"
  CONFIG_PATH="${config_path}" TASK_NAME="${task_name}" python - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
suite = config["paper_grade_full_suite"]
profile_name = suite["entrypoint_registry"][os.environ["TASK_NAME"]]["resource_profile"]
profile = suite["cluster"][profile_name]
for key in ["partition", "nodelist", "cpus_per_task", "mem", "gres"]:
    print(f"{key}={profile.get(key, '')}")
PY
}

entrypoint_build_python_args() {
  local task_name="$1"
  PYTHON_ARGS=("--config" "${CONFIG_PATH}" "--suite-config" "${SUITE_CONFIG_PATH}" "--task" "${task_name}" "--model-family" "${MODEL_FAMILY_VALUE}" "--freeze-manifest" "${FREEZE_MANIFEST_VALUE}")
  if [[ -n "${SEED_VALUE}" ]]; then
    PYTHON_ARGS+=("--seed" "${SEED_VALUE}")
  fi
  if [[ -n "${EXPERIMENT_FAMILY_VALUE}" ]]; then
    PYTHON_ARGS+=("--experiment-family" "${EXPERIMENT_FAMILY_VALUE}")
  fi
  if [[ -n "${OUTPUT_DIR_VALUE}" ]]; then
    PYTHON_ARGS+=("--output-dir" "${OUTPUT_DIR_VALUE}")
  fi
  if [[ -n "${RUN_ID_VALUE}" ]]; then
    PYTHON_ARGS+=("--run-id" "${RUN_ID_VALUE}")
  fi
  if [[ -n "${CHECKPOINT_VALUE}" ]]; then
    PYTHON_ARGS+=("--checkpoint" "${CHECKPOINT_VALUE}")
  fi
  if [[ "${DRY_RUN_VALUE}" -eq 1 ]]; then
    PYTHON_ARGS+=("--dry-run")
  fi
  if [[ -n "${EXTRA_ARGS_VALUE}" ]]; then
    read -r -a EXTRA_ARG_ARRAY <<< "${EXTRA_ARGS_VALUE}"
    PYTHON_ARGS+=("${EXTRA_ARG_ARRAY[@]}")
  fi
}

entrypoint_print_resolved_submission() {
  printf 'RESOLVED_CONFIG_PATH=%s\n' "${CONFIG_PATH}"
  printf 'SUITE_CONFIG_PATH=%s\n' "${SUITE_CONFIG_PATH}"
  printf 'TASK=%s\n' "${TASK_NAME_RESOLVED}"
  printf 'MODEL_FAMILY=%s\n' "${MODEL_FAMILY_VALUE}"
  printf 'EXPERIMENT_FAMILY=%s\n' "${EXPERIMENT_FAMILY_VALUE}"
  printf 'SEED=%s\n' "${SEED_VALUE}"
  printf 'SLURM_JOB_NAME=%s\n' "${JOB_NAME_RESOLVED}"
  printf 'LOG_PATH=%s\n' "${LOG_PATH_RESOLVED}"
  printf 'PYTHON_COMMAND='
  printf '%q ' "${RUN_COMMAND[@]}"
  printf '\n'
  printf 'SBATCH_COMMAND='
  printf '%q ' "${SBATCH_COMMAND[@]}"
  printf '\n'
}

entrypoint_run_task() {
  local default_task_name="$1"
  shift
  entrypoint_parse_common "$@"
  local repo_root
  repo_root="$(entrypoint_repo_root)"
  cd "${repo_root}"
  entrypoint_check_project_prefix "${repo_root}"
  local task_name="${TASK_VALUE:-${default_task_name}}"
  TASK_NAME_RESOLVED="${task_name}"
  local python_path
  python_path="$(entrypoint_registry_value "${CONFIG_PATH}" "${task_name}" "python_path")"
  local python_bin
  python_bin="$(entrypoint_python_bin "${CONFIG_PATH}")"
  entrypoint_build_python_args "${task_name}"
  RUN_COMMAND=("${python_bin}" -u "${python_path}" "${PYTHON_ARGS[@]}")
  declare -A RESOURCE=()
  while IFS='=' read -r key value; do
    RESOURCE["${key}"]="${value}"
  done < <(entrypoint_resource_values "${CONFIG_PATH}" "${task_name}")
  if [[ -n "${PARTITION_VALUE}" ]]; then RESOURCE[partition]="${PARTITION_VALUE}"; fi
  if [[ -n "${NODELIST_VALUE}" ]]; then RESOURCE[nodelist]="${NODELIST_VALUE}"; fi
  if [[ -n "${MEM_VALUE}" ]]; then RESOURCE[mem]="${MEM_VALUE}"; fi
  if [[ -n "${GRES_VALUE}" ]]; then RESOURCE[gres]="${GRES_VALUE}"; fi
  local job_prefix="${JOB_NAME_PREFIX_VALUE:-${PROJECT_PREFIX}}"
  if [[ "${job_prefix}" != "${PROJECT_PREFIX}" && "${job_prefix}" != "${PROJECT_PREFIX}_"* ]]; then
    job_prefix="${PROJECT_PREFIX}_${job_prefix}"
  fi
  JOB_NAME_RESOLVED="${job_prefix}_${task_name}"
  LOG_PATH_RESOLVED="Output/logs/${JOB_NAME_RESOLVED}"
  mkdir -p Output/logs
  local wrapped_command
  wrapped_command="$(printf '%q ' "${RUN_COMMAND[@]}")"
  SBATCH_COMMAND=(
    sbatch
    "--partition=${RESOURCE[partition]}"
    "--nodelist=${RESOURCE[nodelist]}"
    "--cpus-per-task=${RESOURCE[cpus_per_task]}"
    "--mem=${RESOURCE[mem]}"
    "--job-name=${JOB_NAME_RESOLVED}"
    "--output=${LOG_PATH_RESOLVED}.%j.out"
    "--error=${LOG_PATH_RESOLVED}.%j.err"
  )
  if [[ -n "${TIME_VALUE}" ]]; then
    SBATCH_COMMAND+=("--time=${TIME_VALUE}")
  fi
  if [[ -n "${RESOURCE[gres]:-}" ]]; then
    SBATCH_COMMAND+=("--gres=${RESOURCE[gres]}")
  fi
  if [[ -n "${DEPENDENCY_VALUE}" ]]; then
    SBATCH_COMMAND+=("--dependency=${DEPENDENCY_VALUE}")
  fi
  if [[ -n "${DEPENDENCY_JOB_ID_VALUE}" ]]; then
    SBATCH_COMMAND+=("--dependency=afterok:${DEPENDENCY_JOB_ID_VALUE}")
  fi
  SBATCH_COMMAND+=("--wrap=${wrapped_command}")
  entrypoint_print_resolved_submission
  if [[ "${PRINT_ONLY_VALUE}" -eq 1 || "${DRY_RUN_VALUE}" -eq 1 || "${NO_SUBMIT_VALUE}" -eq 1 ]]; then
    return 0
  fi
  bash Libs/scripts/validators/run_prelaunch_gates.sh --config "${CONFIG_PATH}"
  "${SBATCH_COMMAND[@]}"
}
