#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <profile-name> [--print-only] [--config <config-path>] [--dependency-job-id <job-id>]"
  exit 1
fi

PROFILE_NAME="$1"
shift
DRY_RUN=0
CONFIG_PATH="Libs/configs/config_stack.yaml"
UPSTREAM_DEPENDENCY_JOB_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    "")
      shift
      ;;
    --print-only)
      DRY_RUN=1
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
cd "${REPO_ROOT}"

sinfo

PYTHON_BIN="$(CONFIG_PATH="${CONFIG_PATH}" python - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
print(str(config["execution_environment"]["python_executable"]))
PY
)"

readarray -t PROFILE_VALUES < <(CONFIG_PATH="${CONFIG_PATH}" "${PYTHON_BIN}" - "${PROFILE_NAME}" <<'PY'
import sys
import os

from Libs.utils.exp.experiment_manifests import load_experiment_family
from Libs.utils.data.raw_data_registry import load_config

profile_name = sys.argv[1]
config = load_config(os.environ["CONFIG_PATH"])
profile = load_experiment_family(profile_name, config)
runtime = profile.get("runtime", {})
cluster = profile.get("cluster_training", profile.get("cluster", {}))
execution = profile.get("execution", {})

def emit(prefix, payload):
    for key, value in payload.items():
        print(f"{prefix}::{key}::{value}")

emit("cluster", cluster)
print(f"runtime::device::{runtime.get('device', 'cpu')}")
print(f"runtime::training_scope::{runtime.get('training_scope', 'all')}")
print(f"runtime::disease_program::{profile.get('disease_program', '')}")
print(f"freeze::sha256::{config.get('paper_grade_full_suite', {}).get('freeze_snapshot_signature_sha256', '')}")
for seed_value in runtime.get("seed_values", []):
    print(f"seed::{seed_value}")
for command in execution.get("train_commands", []):
    print(f"train_command::{command}")
PY
)

PROJECT_PREFIX="$(cat PROJECT_PREFIX)"
CONFIG_OUTPUT_ROOT="Output/paper_configs"
SBATCH_OUTPUT_ROOT="Output/logs"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  CONFIG_OUTPUT_ROOT="Output/paper_configs/print_only/${PROFILE_NAME}"
  SBATCH_OUTPUT_ROOT="Output/logs/print_only/${PROFILE_NAME}"
fi
mkdir -p "${CONFIG_OUTPUT_ROOT}" "${SBATCH_OUTPUT_ROOT}"
declare -A CLUSTER=()
DEVICE_NAME="cpu"
TRAINING_SCOPE="all"
DISEASE_PROGRAM=""
FREEZE_SIGNATURE_SHA256=""
SEED_VALUES=()
TRAIN_COMMANDS=()
for value in "${PROFILE_VALUES[@]}"; do
  if [[ "${value}" == cluster::* ]]; then
    KEY="${value#cluster::}"
    KEY="${KEY%%::*}"
    VAL="${value##*::}"
    CLUSTER["${KEY}"]="${VAL}"
  elif [[ "${value}" == runtime::device::* ]]; then
    DEVICE_NAME="${value##*::}"
  elif [[ "${value}" == runtime::training_scope::* ]]; then
    TRAINING_SCOPE="${value##*::}"
  elif [[ "${value}" == runtime::disease_program::* ]]; then
    DISEASE_PROGRAM="${value##*::}"
  elif [[ "${value}" == freeze::sha256::* ]]; then
    FREEZE_SIGNATURE_SHA256="${value#freeze::sha256::}"
  elif [[ "${value}" == seed::* ]]; then
    SEED_VALUES+=("${value#seed::}")
  elif [[ "${value}" == train_command::* ]]; then
    TRAIN_COMMANDS+=("${value#train_command::}")
  fi
done

if [[ "${#SEED_VALUES[@]}" -eq 0 ]]; then
  echo "Profile ${PROFILE_NAME} has no configured seed_values."
  exit 1
fi

if [[ "${#TRAIN_COMMANDS[@]}" -eq 0 ]]; then
  echo "Profile ${PROFILE_NAME} has no configured train_commands."
  exit 1
fi

LAST_JOB_ID=""
for SEED_VALUE in "${SEED_VALUES[@]}"; do
  JOB_SUFFIX="${PROFILE_NAME}_seed_${SEED_VALUE}"
  JOB_NAME="${PROJECT_PREFIX}_${JOB_SUFFIX}"
  OUTPUT_CONFIG="${CONFIG_OUTPUT_ROOT}/${JOB_SUFFIX}.yaml"
  SBATCH_PATH="${SBATCH_OUTPUT_ROOT}/${JOB_NAME}.sbatch"

  "${PYTHON_BIN}" Libs/exp/orchestration/prepare_experiment_config.py \
    --profile "${PROFILE_NAME}" \
    --seed "${SEED_VALUE}" \
    --output-config "${OUTPUT_CONFIG}" \
    --base-config "${CONFIG_PATH}"

  GRES_LINE=""
  if [[ -n "${CLUSTER[gres]:-}" ]]; then
    GRES_LINE="#SBATCH --gres=${CLUSTER[gres]}"
  fi

  {
    echo "#!/usr/bin/env bash"
    echo "#SBATCH --partition=${CLUSTER[partition]}"
    echo "#SBATCH --nodelist=${CLUSTER[nodelist]}"
    echo "#SBATCH --cpus-per-task=${CLUSTER[cpus_per_task]}"
    echo "#SBATCH --mem=${CLUSTER[mem]}"
    if [[ -n "${GRES_LINE}" ]]; then
      echo "${GRES_LINE}"
    fi
    echo "#SBATCH --job-name=${JOB_NAME}"
    echo "#SBATCH --output=Output/logs/${JOB_NAME}.%j.out"
    echo "#SBATCH --error=Output/logs/${JOB_NAME}.%j.err"
    echo
    echo "set -euo pipefail"
    if [[ -n "${FREEZE_SIGNATURE_SHA256}" ]]; then
      echo "export PAPER_GRADE_FREEZE_SIGNATURE_SHA256=${FREEZE_SIGNATURE_SHA256}"
    fi
    echo
    echo "cd ${REPO_ROOT}"
    for COMMAND_TEMPLATE in "${TRAIN_COMMANDS[@]}"; do
      COMMAND="${COMMAND_TEMPLATE//\{config_path\}/${OUTPUT_CONFIG}}"
      COMMAND="${COMMAND//\{training_scope\}/${TRAINING_SCOPE}}"
      COMMAND="${COMMAND//\{device\}/${DEVICE_NAME}}"
      COMMAND="${COMMAND//\{disease_program\}/${DISEASE_PROGRAM}}"
      COMMAND="${COMMAND//\{python_executable\}/${PYTHON_BIN}}"
      echo "${COMMAND}"
    done
  } > "${SBATCH_PATH}"

  chmod +x "${SBATCH_PATH}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Prepared ${SBATCH_PATH}"
    continue
  fi

  SBATCH_ARGS=()
  DEPENDENCY_IDS=()
  if [[ -n "${UPSTREAM_DEPENDENCY_JOB_ID}" ]]; then
    DEPENDENCY_IDS+=("${UPSTREAM_DEPENDENCY_JOB_ID}")
  fi
  if [[ -n "${LAST_JOB_ID}" ]]; then
    DEPENDENCY_IDS+=("${LAST_JOB_ID}")
  fi
  if [[ "${#DEPENDENCY_IDS[@]}" -gt 0 ]]; then
    DEPENDENCY_VALUE="$(IFS=:; echo "${DEPENDENCY_IDS[*]}")"
    SBATCH_ARGS+=("--dependency=afterok:${DEPENDENCY_VALUE}")
  fi
  LAST_JOB_ID="$(sbatch --parsable "${SBATCH_ARGS[@]}" "${SBATCH_PATH}")"
done

if [[ "${DRY_RUN}" -eq 0 && -n "${LAST_JOB_ID}" ]]; then
  echo "PROFILE_LAST_JOB_ID=${LAST_JOB_ID}"
fi
