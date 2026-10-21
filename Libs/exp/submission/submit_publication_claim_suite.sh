#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="Libs/configs/config_stack.yaml"
PRINT_ONLY=0
UPSTREAM_DEPENDENCY_JOB_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    "")
      shift
      ;;
    --print-only)
      PRINT_ONLY=1
      shift
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --dependency-job-id)
      UPSTREAM_DEPENDENCY_JOB_ID="$2"
      shift 2
      ;;
    *)
      CONFIG_PATH="$1"
      shift
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

sinfo

PYTHON_BIN="$(CONFIG_PATH="${CONFIG_PATH}" python - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
print(str(config["execution_environment"]["python_executable"]))
PY
)"

readarray -t SUITE_VALUES < <(CONFIG_PATH="${CONFIG_PATH}" "${PYTHON_BIN}" - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
suite = config["paper"]["publication_claim_suite"]
cluster_profile = str(suite["final_report_cluster_profile"])
cluster = config["cluster_registry"]["active_learning"][cluster_profile]
for profile_name in suite["dynamics_profiles"]:
    print(f"dynamics::{profile_name}")
for profile_name in suite.get("secondary_benchmark_profiles", []):
    print(f"secondary::{profile_name}")
print(f"suite::baseline_suite_enabled::{suite['baseline_suite_enabled']}")
print(f"suite::framework_audit_enabled::{suite.get('framework_audit_enabled', False)}")
print(f"suite::fsdp_smoke_enabled::{suite['fsdp_smoke_enabled']}")
print(f"suite::final_report_job_suffix::{suite['final_report_job_suffix']}")
for key, value in cluster.items():
    print(f"cluster::{key}::{value}")
for command in suite["final_report_commands"]:
    print(f"final_command::{command}")
PY
)

DYNAMICS_PROFILES=()
SECONDARY_PROFILES=()
BASELINE_SUITE_ENABLED=0
FRAMEWORK_AUDIT_ENABLED=0
FSDP_SMOKE_ENABLED=0
FINAL_REPORT_JOB_SUFFIX=""
FINAL_COMMANDS=()
declare -A FINAL_CLUSTER=()
for value in "${SUITE_VALUES[@]}"; do
  if [[ "${value}" == dynamics::* ]]; then
    DYNAMICS_PROFILES+=("${value#dynamics::}")
  elif [[ "${value}" == secondary::* ]]; then
    SECONDARY_PROFILES+=("${value#secondary::}")
  elif [[ "${value}" == suite::baseline_suite_enabled::* ]]; then
    if [[ "${value##*::}" == "True" || "${value##*::}" == "true" ]]; then
      BASELINE_SUITE_ENABLED=1
    fi
  elif [[ "${value}" == suite::framework_audit_enabled::* ]]; then
    if [[ "${value##*::}" == "True" || "${value##*::}" == "true" ]]; then
      FRAMEWORK_AUDIT_ENABLED=1
    fi
  elif [[ "${value}" == suite::fsdp_smoke_enabled::* ]]; then
    if [[ "${value##*::}" == "True" || "${value##*::}" == "true" ]]; then
      FSDP_SMOKE_ENABLED=1
    fi
  elif [[ "${value}" == suite::final_report_job_suffix::* ]]; then
    FINAL_REPORT_JOB_SUFFIX="${value##*::}"
  elif [[ "${value}" == cluster::* ]]; then
    KEY="${value#cluster::}"
    KEY="${KEY%%::*}"
    VAL="${value##*::}"
    FINAL_CLUSTER["${KEY}"]="${VAL}"
  elif [[ "${value}" == final_command::* ]]; then
    FINAL_COMMANDS+=("${value#final_command::}")
  fi
done

if [[ "${#DYNAMICS_PROFILES[@]}" -eq 0 ]]; then
  echo "publication_claim_suite.dynamics_profiles is empty."
  exit 1
fi

PRINT_ONLY_ARG=()
if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  PRINT_ONLY_ARG+=("--print-only")
fi

DEPENDENCY_ARG=()
if [[ -n "${UPSTREAM_DEPENDENCY_JOB_ID}" ]]; then
  DEPENDENCY_ARG+=("--dependency-job-id" "${UPSTREAM_DEPENDENCY_JOB_ID}")
fi

LAST_DYNAMICS_JOB_ID=""
submit_dynamics_profile() {
  local profile_name="$1"
  local output
  output="$(bash "${SCRIPT_DIR}/submit_dynamics_profile_suite.sh" "${profile_name}" --config "${CONFIG_PATH}" "${PRINT_ONLY_ARG[@]}" "${DEPENDENCY_ARG[@]}")"
  printf '%s\n' "${output}"
  if [[ "${PRINT_ONLY}" -eq 0 ]]; then
    LAST_DYNAMICS_JOB_ID="$(printf '%s\n' "${output}" | awk -F= '$1 == "PROFILE_LAST_JOB_ID" {print $2}' | tail -n 1)"
    if [[ -z "${LAST_DYNAMICS_JOB_ID}" ]]; then
      echo "Failed to parse PROFILE_LAST_JOB_ID for ${profile_name}."
      exit 1
    fi
    DEPENDENCY_ARG=("--dependency-job-id" "${LAST_DYNAMICS_JOB_ID}")
  fi
}

reuse_active_profile_lock() {
  local profile_name="$1"
  if [[ "${PRINT_ONLY}" -ne 0 ]]; then
    return 1
  fi
  local lock_path="Output/locks/dynamics_profile_${profile_name}_last_job.txt"
  if [[ ! -f "${lock_path}" ]]; then
    return 1
  fi
  local candidate_job_id
  candidate_job_id="$(cat "${lock_path}")"
  if [[ -z "${candidate_job_id}" ]]; then
    rm -f "${lock_path}"
    return 1
  fi
  local active_job_id
  active_job_id="$(squeue -h -j "${candidate_job_id}" -o "%i" 2>/dev/null | head -n 1 || true)"
  if [[ -z "${active_job_id}" ]]; then
    rm -f "${lock_path}"
    return 1
  fi
  LAST_DYNAMICS_JOB_ID="${candidate_job_id}"
  DEPENDENCY_ARG=("--dependency-job-id" "${LAST_DYNAMICS_JOB_ID}")
  echo "PROFILE_REUSED_ACTIVE_JOB_ID=${profile_name}:${LAST_DYNAMICS_JOB_ID}"
  return 0
}

if [[ "${PRINT_ONLY}" -eq 0 ]]; then
  "${PYTHON_BIN}" -u Libs/scripts/validators/validate_simulation_readiness_audit.py --config "${CONFIG_PATH}"
  "${PYTHON_BIN}" -u Libs/scripts/validators/validate_supplemental_milestone_contracts.py --config "${CONFIG_PATH}"
  "${PYTHON_BIN}" -u Libs/scripts/validators/validate_formal_experiment_readiness.py --config "${CONFIG_PATH}"
  "${PYTHON_BIN}" -u Libs/scripts/validators/validate_formal_experiment_freeze.py --config "${CONFIG_PATH}" --no-write
fi

for profile_name in "${DYNAMICS_PROFILES[@]}" "${SECONDARY_PROFILES[@]}"; do
  if reuse_active_profile_lock "${profile_name}"; then
    continue
  fi
  submit_dynamics_profile "${profile_name}"
done

BASELINE_JOB_IDS=""
if [[ "${BASELINE_SUITE_ENABLED}" -eq 1 ]]; then
  BASELINE_ARGS=("--config" "${CONFIG_PATH}" "${PRINT_ONLY_ARG[@]}")
  if [[ "${PRINT_ONLY}" -eq 0 && -n "${LAST_DYNAMICS_JOB_ID}" ]]; then
    BASELINE_ARGS+=("--dependency-job-id" "${LAST_DYNAMICS_JOB_ID}")
  elif [[ "${PRINT_ONLY}" -eq 0 && -n "${UPSTREAM_DEPENDENCY_JOB_ID}" ]]; then
    BASELINE_ARGS+=("--dependency-job-id" "${UPSTREAM_DEPENDENCY_JOB_ID}")
  fi
  BASELINE_OUTPUT="$(bash "${SCRIPT_DIR}/submit_runnable_baseline_suite.sh" "${BASELINE_ARGS[@]}")"
  printf '%s\n' "${BASELINE_OUTPUT}"
  if [[ "${PRINT_ONLY}" -eq 0 ]]; then
    BASELINE_JOB_IDS="$(printf '%s\n' "${BASELINE_OUTPUT}" | awk -F= '$1 == "BASELINE_JOB_IDS" {print $2}' | tail -n 1)"
  fi
fi

AUDIT_JOB_ID=""
if [[ "${FRAMEWORK_AUDIT_ENABLED}" -eq 1 ]]; then
  AUDIT_ARGS=("--config" "${CONFIG_PATH}" "${PRINT_ONLY_ARG[@]}")
  if [[ "${PRINT_ONLY}" -eq 0 && -n "${LAST_DYNAMICS_JOB_ID}" ]]; then
    AUDIT_ARGS+=("--dependency-job-id" "${LAST_DYNAMICS_JOB_ID}")
  elif [[ "${PRINT_ONLY}" -eq 0 && -n "${UPSTREAM_DEPENDENCY_JOB_ID}" ]]; then
    AUDIT_ARGS+=("--dependency-job-id" "${UPSTREAM_DEPENDENCY_JOB_ID}")
  fi
  AUDIT_OUTPUT="$(bash "${SCRIPT_DIR}/submit_framework_hardening_suite.sh" "${AUDIT_ARGS[@]}")"
  printf '%s\n' "${AUDIT_OUTPUT}"
  if [[ "${PRINT_ONLY}" -eq 0 ]]; then
    AUDIT_JOB_ID="$(printf '%s\n' "${AUDIT_OUTPUT}" | awk -F= '$1 == "AUDIT_LAST_JOB_ID" {print $2}' | tail -n 1)"
  fi
fi

FSDP_JOB_ID=""
if [[ "${FSDP_SMOKE_ENABLED}" -eq 1 ]]; then
  FSDP_ARGS=("--config" "${CONFIG_PATH}" "${PRINT_ONLY_ARG[@]}")
  if [[ "${PRINT_ONLY}" -eq 0 && -n "${LAST_DYNAMICS_JOB_ID}" ]]; then
    FSDP_ARGS+=("--dependency-job-id" "${LAST_DYNAMICS_JOB_ID}")
  elif [[ "${PRINT_ONLY}" -eq 0 && -n "${UPSTREAM_DEPENDENCY_JOB_ID}" ]]; then
    FSDP_ARGS+=("--dependency-job-id" "${UPSTREAM_DEPENDENCY_JOB_ID}")
  fi
  FSDP_OUTPUT="$(bash "${SCRIPT_DIR}/submit_neuronsifter_fsdp_smoke.sh" "${FSDP_ARGS[@]}")"
  printf '%s\n' "${FSDP_OUTPUT}"
  if [[ "${PRINT_ONLY}" -eq 0 ]]; then
    FSDP_JOB_ID="$(printf '%s\n' "${FSDP_OUTPUT}" | awk -F= '$1 == "FSDP_JOB_ID" {print $2}' | tail -n 1)"
  fi
fi

PROJECT_PREFIX="$(cat PROJECT_PREFIX)"
FINAL_REPORT_JOB_NAME="${PROJECT_PREFIX}_${FINAL_REPORT_JOB_SUFFIX}"
SBATCH_OUTPUT_ROOT="Output/logs"
if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  SBATCH_OUTPUT_ROOT="Output/logs/print_only/publication_claim_suite"
fi
mkdir -p "${SBATCH_OUTPUT_ROOT}"
FINAL_REPORT_SBATCH="${SBATCH_OUTPUT_ROOT}/${FINAL_REPORT_JOB_NAME}.sbatch"

{
  echo "#!/usr/bin/env bash"
  echo "#SBATCH --partition=${FINAL_CLUSTER[partition]}"
  echo "#SBATCH --nodelist=${FINAL_CLUSTER[nodelist]}"
  echo "#SBATCH --cpus-per-task=${FINAL_CLUSTER[cpus_per_task]}"
  echo "#SBATCH --mem=${FINAL_CLUSTER[mem]}"
  if [[ -n "${FINAL_CLUSTER[gres]:-}" ]]; then
    echo "#SBATCH --gres=${FINAL_CLUSTER[gres]}"
  fi
  echo "#SBATCH --job-name=${FINAL_REPORT_JOB_NAME}"
  echo "#SBATCH --output=Output/logs/${FINAL_REPORT_JOB_NAME}.%j.out"
  echo "#SBATCH --error=Output/logs/${FINAL_REPORT_JOB_NAME}.%j.err"
  echo
  echo "set -euo pipefail"
  echo "cd ${REPO_ROOT}"
  for command_template in "${FINAL_COMMANDS[@]}"; do
    command="${command_template//\{config_path\}/${CONFIG_PATH}}"
    command="${command//\{python_executable\}/${PYTHON_BIN}}"
    echo "${command}"
  done
} > "${FINAL_REPORT_SBATCH}"
chmod +x "${FINAL_REPORT_SBATCH}"

if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  echo "Prepared ${FINAL_REPORT_SBATCH}"
  exit 0
fi

FINAL_DEPENDENCIES=()
if [[ -n "${LAST_DYNAMICS_JOB_ID}" ]]; then
  FINAL_DEPENDENCIES+=("${LAST_DYNAMICS_JOB_ID}")
fi
if [[ -n "${BASELINE_JOB_IDS}" ]]; then
  FINAL_DEPENDENCIES+=("${BASELINE_JOB_IDS}")
fi
if [[ -n "${FSDP_JOB_ID}" ]]; then
  FINAL_DEPENDENCIES+=("${FSDP_JOB_ID}")
fi
if [[ -n "${AUDIT_JOB_ID}" ]]; then
  FINAL_DEPENDENCIES+=("${AUDIT_JOB_ID}")
fi
if [[ "${#FINAL_DEPENDENCIES[@]}" -gt 0 ]]; then
  FINAL_DEPENDENCY_VALUE="$(IFS=:; echo "${FINAL_DEPENDENCIES[*]}")"
  FINAL_REPORT_JOB_ID="$(sbatch --parsable --dependency=afterok:${FINAL_DEPENDENCY_VALUE} "${FINAL_REPORT_SBATCH}")"
else
  FINAL_REPORT_JOB_ID="$(sbatch --parsable "${FINAL_REPORT_SBATCH}")"
fi
echo "FINAL_REPORT_JOB_ID=${FINAL_REPORT_JOB_ID}"
