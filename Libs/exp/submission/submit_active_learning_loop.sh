#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-Libs/configs/config_stack.yaml}"
PRINT_ONLY=0
RUN_MATERIALIZATION=0
for arg in "${@:2}"; do
  if [[ "${arg}" == "--print-only" ]]; then
    PRINT_ONLY=1
  fi
  if [[ "${arg}" == "--materialize" ]]; then
    RUN_MATERIALIZATION=1
  fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

sinfo

PYTHON_BIN="$(CONFIG_PATH="${CONFIG_PATH}" python - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
print(str(config["execution_environment"]["python_executable"]))
PY
)"

readarray -t ACTIVE_CLUSTER_VALUES < <(CONFIG_PATH="${CONFIG_PATH}" "${PYTHON_BIN}" - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
for profile_name, profile in config["cluster_registry"]["active_learning"].items():
    for key, value in profile.items():
        print(f"{profile_name}::{key}::{value}")
PY
)

readarray -t ACTIVE_ROUNDS < <(CONFIG_PATH="${CONFIG_PATH}" "${PYTHON_BIN}" - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
for row in config["screening_pipeline"]["screening_benchmark"]["active_learning"]["active_rounds"]:
    print(str(row["round_uid"]))
PY
)

cluster_value() {
  local profile_name="$1"
  local key_name="$2"
  local row
  for row in "${ACTIVE_CLUSTER_VALUES[@]}"; do
    local row_profile="${row%%::*}"
    local remainder="${row#*::}"
    local row_key="${remainder%%::*}"
    local row_value="${remainder##*::}"
    if [[ "${row_profile}" == "${profile_name}" && "${row_key}" == "${key_name}" ]]; then
      printf '%s\n' "${row_value}"
      return 0
    fi
  done
  return 1
}

write_job() {
  local profile_name="$1"
  local job_name="$2"
  local sbatch_path="$3"
  shift 3
  {
    echo "#!/usr/bin/env bash"
    echo "#SBATCH --partition=$(cluster_value "${profile_name}" partition)"
    echo "#SBATCH --nodelist=$(cluster_value "${profile_name}" nodelist)"
    echo "#SBATCH --cpus-per-task=$(cluster_value "${profile_name}" cpus_per_task)"
    echo "#SBATCH --mem=$(cluster_value "${profile_name}" mem)"
    local gres_value
    gres_value="$(cluster_value "${profile_name}" gres || true)"
    if [[ -n "${gres_value}" ]]; then
      echo "#SBATCH --gres=${gres_value}"
    fi
    echo "#SBATCH --job-name=${job_name}"
    echo "#SBATCH --output=Output/logs/${job_name}.%j.out"
    echo "#SBATCH --error=Output/logs/${job_name}.%j.err"
    echo
    echo "set -euo pipefail"
    echo "cd ${REPO_ROOT}"
    local command
    for command in "$@"; do
      echo "${command}"
    done
  } > "${sbatch_path}"
  chmod +x "${sbatch_path}"
}

SBATCH_OUTPUT_ROOT="Output/logs"
if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  SBATCH_OUTPUT_ROOT="Output/logs/print_only/active_learning"
fi
mkdir -p "${SBATCH_OUTPUT_ROOT}"

PROJECT_PREFIX="$(cat PROJECT_PREFIX)"
MATERIALIZATION_JOB_NAME="${PROJECT_PREFIX}_external_adapter_materialization"
MATERIALIZATION_SBATCH_PATH="${SBATCH_OUTPUT_ROOT}/${MATERIALIZATION_JOB_NAME}.sbatch"
write_job "external_adapter_materialization" "${MATERIALIZATION_JOB_NAME}" "${MATERIALIZATION_SBATCH_PATH}" \
  "${PYTHON_BIN} -u Libs/scripts/builders/build_external_adapter_processed_catalogs.py --config ${CONFIG_PATH}" \
  "${PYTHON_BIN} -u Libs/scripts/builders/build_external_adapter_registry.py --config ${CONFIG_PATH}" \
  "${PYTHON_BIN} -u Libs/scripts/validators/validate_external_adapter_registry.py --config ${CONFIG_PATH} --strict-processed"

CPU_TEST_JOB_NAME="${PROJECT_PREFIX}_al_cpu_tests"
CPU_TEST_SBATCH_PATH="${SBATCH_OUTPUT_ROOT}/${CPU_TEST_JOB_NAME}.sbatch"
write_job "cpu_tests" "${CPU_TEST_JOB_NAME}" "${CPU_TEST_SBATCH_PATH}" \
  "${PYTHON_BIN} -m pytest Libs/tests -q"

PREFLIGHT_JOB_NAME="${PROJECT_PREFIX}_al_preflight"
PREFLIGHT_SBATCH_PATH="${SBATCH_OUTPUT_ROOT}/${PREFLIGHT_JOB_NAME}.sbatch"
write_job "preflight" "${PREFLIGHT_JOB_NAME}" "${PREFLIGHT_SBATCH_PATH}" \
  "${PYTHON_BIN} -u Libs/scripts/active_learning/preflight_active_learning_loop.py --config ${CONFIG_PATH} --strict-processed --strict-binaries"

declare -a ROUND_REQUEST_SCRIPTS=()
declare -a ROUND_CANDIDATE_PREPARATION_SCRIPTS=()
declare -a ROUND_CANDIDATE_DOCKING_SCRIPTS=()
declare -a ROUND_CANDIDATE_REGISTRY_SCRIPTS=()
declare -a ROUND_SIMULATOR_SCRIPTS=()
declare -a ROUND_UPDATE_SCRIPTS=()
declare -a ROUND_VALIDATE_SCRIPTS=()

for round_uid in "${ACTIVE_ROUNDS[@]}"; do
  REQUEST_JOB_NAME="${PROJECT_PREFIX}_${round_uid}_request"
  REQUEST_SBATCH_PATH="${SBATCH_OUTPUT_ROOT}/${REQUEST_JOB_NAME}.sbatch"
  write_job "round_cpu" "${REQUEST_JOB_NAME}" "${REQUEST_SBATCH_PATH}" \
    "${PYTHON_BIN} -u Libs/scripts/active_learning/build_active_learning_priority_catalog.py --config ${CONFIG_PATH}" \
    "${PYTHON_BIN} -u Libs/exp/active_learning/run_active_learning_round.py --config ${CONFIG_PATH} --round-uid ${round_uid}" \
    "${PYTHON_BIN} -u Libs/scripts/active_learning/validate_active_learning_round.py --config ${CONFIG_PATH} --round-uid ${round_uid}"
  ROUND_REQUEST_SCRIPTS+=("${REQUEST_SBATCH_PATH}")

  CANDIDATE_PREPARATION_JOB_NAME="${PROJECT_PREFIX}_${round_uid}_candidate_preparation"
  CANDIDATE_PREPARATION_SBATCH_PATH="${SBATCH_OUTPUT_ROOT}/${CANDIDATE_PREPARATION_JOB_NAME}.sbatch"
  write_job "round_cpu" "${CANDIDATE_PREPARATION_JOB_NAME}" "${CANDIDATE_PREPARATION_SBATCH_PATH}" \
    "${PYTHON_BIN} -u Libs/scripts/builders/build_candidate_validation_registry.py --config ${CONFIG_PATH}" \
    "${PYTHON_BIN} -u Libs/exp/execution/run_candidate_validation_execution.py --config ${CONFIG_PATH} --execution-stage prep"
  ROUND_CANDIDATE_PREPARATION_SCRIPTS+=("${CANDIDATE_PREPARATION_SBATCH_PATH}")

  CANDIDATE_DOCKING_JOB_NAME="${PROJECT_PREFIX}_${round_uid}_candidate_docking"
  CANDIDATE_DOCKING_SBATCH_PATH="${SBATCH_OUTPUT_ROOT}/${CANDIDATE_DOCKING_JOB_NAME}.sbatch"
  write_job "round_gpu" "${CANDIDATE_DOCKING_JOB_NAME}" "${CANDIDATE_DOCKING_SBATCH_PATH}" \
    "${PYTHON_BIN} -u Libs/exp/execution/run_candidate_validation_execution.py --config ${CONFIG_PATH} --execution-stage docking"
  ROUND_CANDIDATE_DOCKING_SCRIPTS+=("${CANDIDATE_DOCKING_SBATCH_PATH}")

  CANDIDATE_REGISTRY_JOB_NAME="${PROJECT_PREFIX}_${round_uid}_candidate_registry"
  CANDIDATE_REGISTRY_SBATCH_PATH="${SBATCH_OUTPUT_ROOT}/${CANDIDATE_REGISTRY_JOB_NAME}.sbatch"
  write_job "round_cpu" "${CANDIDATE_REGISTRY_JOB_NAME}" "${CANDIDATE_REGISTRY_SBATCH_PATH}" \
    "${PYTHON_BIN} -u Libs/scripts/builders/build_candidate_validation_registry.py --config ${CONFIG_PATH}" \
    "${PYTHON_BIN} -u Libs/scripts/validators/validate_candidate_validation_registry.py --config ${CONFIG_PATH}"
  ROUND_CANDIDATE_REGISTRY_SCRIPTS+=("${CANDIDATE_REGISTRY_SBATCH_PATH}")

  SIMULATOR_JOB_NAME="${PROJECT_PREFIX}_${round_uid}_teacher_integration"
  SIMULATOR_SBATCH_PATH="${SBATCH_OUTPUT_ROOT}/${SIMULATOR_JOB_NAME}.sbatch"
  write_job "round_simulator_cpu" "${SIMULATOR_JOB_NAME}" "${SIMULATOR_SBATCH_PATH}" \
    "${PYTHON_BIN} -u Libs/scripts/builders/build_external_adapter_processed_catalogs.py --config ${CONFIG_PATH} --source moleculenet_deepchem" \
    "${PYTHON_BIN} -u Libs/scripts/teacher_generation/build_osp_pbpk_teacher_rollouts.py --config ${CONFIG_PATH}" \
    "${PYTHON_BIN} -u Libs/scripts/teacher_generation/run_neuron_rxd_microenvironment_teacher.py --config ${CONFIG_PATH}" \
    "${PYTHON_BIN} -u Libs/scripts/builders/build_full_microenvironment_teacher_registry.py --config ${CONFIG_PATH}" \
    "${PYTHON_BIN} -u Libs/scripts/active_learning/link_completed_simulation_results.py --config ${CONFIG_PATH} --round-uid ${round_uid}" \
    "${PYTHON_BIN} -u Libs/scripts/active_learning/ingest_simulation_results.py --config ${CONFIG_PATH} --round-uid ${round_uid}" \
    "${PYTHON_BIN} -u Libs/scripts/active_learning/build_active_learning_round_report.py --config ${CONFIG_PATH} --round-uid ${round_uid}"
  ROUND_SIMULATOR_SCRIPTS+=("${SIMULATOR_SBATCH_PATH}")

  UPDATE_JOB_NAME="${PROJECT_PREFIX}_${round_uid}_model_update"
  UPDATE_SBATCH_PATH="${SBATCH_OUTPUT_ROOT}/${UPDATE_JOB_NAME}.sbatch"
  write_job "round_gpu" "${UPDATE_JOB_NAME}" "${UPDATE_SBATCH_PATH}" \
    "${PYTHON_BIN} -u Libs/exp/active_learning/run_active_learning_model_update.py --config ${CONFIG_PATH} --round-uid ${round_uid} --device cuda"
  ROUND_UPDATE_SCRIPTS+=("${UPDATE_SBATCH_PATH}")

  VALIDATE_JOB_NAME="${PROJECT_PREFIX}_${round_uid}_strict_holdout"
  VALIDATE_SBATCH_PATH="${SBATCH_OUTPUT_ROOT}/${VALIDATE_JOB_NAME}.sbatch"
  write_job "round_cpu" "${VALIDATE_JOB_NAME}" "${VALIDATE_SBATCH_PATH}" \
    "${PYTHON_BIN} -u Libs/scripts/active_learning/validate_active_learning_round.py --config ${CONFIG_PATH} --round-uid ${round_uid} --require-results"
  ROUND_VALIDATE_SCRIPTS+=("${VALIDATE_SBATCH_PATH}")
done

PUBLICATION_JOB_NAME="${PROJECT_PREFIX}_publication_evidence"
PUBLICATION_SBATCH_PATH="${SBATCH_OUTPUT_ROOT}/${PUBLICATION_JOB_NAME}.sbatch"
PUBLICATION_COMMANDS=(
  "${PYTHON_BIN} -u Libs/exp/active_learning/evaluate_active_learning_strict_holdouts.py --config ${CONFIG_PATH}"
)
write_job "publication" "${PUBLICATION_JOB_NAME}" "${PUBLICATION_SBATCH_PATH}" "${PUBLICATION_COMMANDS[@]}"

FOUNDATION_JOB_NAME="${PROJECT_PREFIX}_foundation_sample_audit"
FOUNDATION_SBATCH_PATH="${SBATCH_OUTPUT_ROOT}/${FOUNDATION_JOB_NAME}.sbatch"
write_job "g5_preprocessing" "${FOUNDATION_JOB_NAME}" "${FOUNDATION_SBATCH_PATH}" \
  "${PYTHON_BIN} -u Libs/scripts/builders/build_simulation_readiness_audit.py --config ${CONFIG_PATH}" \
  "${PYTHON_BIN} -u Libs/scripts/validators/validate_simulation_readiness_audit.py --config ${CONFIG_PATH}" \
  "${PYTHON_BIN} -u Libs/scripts/validators/validate_supplemental_milestone_contracts.py --config ${CONFIG_PATH}"

if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  echo "Prepared optional ${MATERIALIZATION_SBATCH_PATH}"
  echo "Prepared ${CPU_TEST_SBATCH_PATH}"
  echo "Prepared ${PREFLIGHT_SBATCH_PATH}"
  for path in "${ROUND_REQUEST_SCRIPTS[@]}" "${ROUND_CANDIDATE_PREPARATION_SCRIPTS[@]}" "${ROUND_CANDIDATE_DOCKING_SCRIPTS[@]}" "${ROUND_CANDIDATE_REGISTRY_SCRIPTS[@]}" "${ROUND_SIMULATOR_SCRIPTS[@]}" "${ROUND_UPDATE_SCRIPTS[@]}" "${ROUND_VALIDATE_SCRIPTS[@]}"; do
    echo "Prepared ${path}"
  done
  echo "Prepared ${PUBLICATION_SBATCH_PATH}"
  echo "Prepared ${FOUNDATION_SBATCH_PATH}"
  exit 0
fi

if [[ "${RUN_MATERIALIZATION}" -eq 1 ]]; then
  sbatch --wait --parsable "${MATERIALIZATION_SBATCH_PATH}"
fi
sbatch --wait --parsable "${CPU_TEST_SBATCH_PATH}"
sbatch --wait --parsable "${PREFLIGHT_SBATCH_PATH}"
PREVIOUS_JOB_ID=""
for index in "${!ACTIVE_ROUNDS[@]}"; do
  if [[ -n "${PREVIOUS_JOB_ID}" ]]; then
    REQUEST_JOB_ID="$(sbatch --dependency=afterok:${PREVIOUS_JOB_ID} --parsable "${ROUND_REQUEST_SCRIPTS[${index}]}")"
  else
    REQUEST_JOB_ID="$(sbatch --parsable "${ROUND_REQUEST_SCRIPTS[${index}]}")"
  fi
  CANDIDATE_PREPARATION_JOB_ID="$(sbatch --dependency=afterok:${REQUEST_JOB_ID} --parsable "${ROUND_CANDIDATE_PREPARATION_SCRIPTS[${index}]}")"
  CANDIDATE_DOCKING_JOB_ID="$(sbatch --dependency=afterok:${CANDIDATE_PREPARATION_JOB_ID} --parsable "${ROUND_CANDIDATE_DOCKING_SCRIPTS[${index}]}")"
  CANDIDATE_REGISTRY_JOB_ID="$(sbatch --dependency=afterok:${CANDIDATE_DOCKING_JOB_ID} --parsable "${ROUND_CANDIDATE_REGISTRY_SCRIPTS[${index}]}")"
  SIMULATOR_JOB_ID="$(sbatch --dependency=afterok:${CANDIDATE_REGISTRY_JOB_ID} --parsable "${ROUND_SIMULATOR_SCRIPTS[${index}]}")"
  UPDATE_JOB_ID="$(sbatch --dependency=afterok:${SIMULATOR_JOB_ID} --parsable "${ROUND_UPDATE_SCRIPTS[${index}]}")"
  VALIDATE_JOB_ID="$(sbatch --dependency=afterok:${UPDATE_JOB_ID} --parsable "${ROUND_VALIDATE_SCRIPTS[${index}]}")"
  PREVIOUS_JOB_ID="${VALIDATE_JOB_ID}"
done
PUBLICATION_JOB_ID="$(sbatch --dependency=afterok:${PREVIOUS_JOB_ID} --parsable "${PUBLICATION_SBATCH_PATH}")"
sbatch --dependency=afterok:${PUBLICATION_JOB_ID} "${FOUNDATION_SBATCH_PATH}"
