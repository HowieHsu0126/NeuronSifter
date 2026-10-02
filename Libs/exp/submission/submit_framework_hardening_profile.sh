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
import os
import sys
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.exp.experiment_manifests import load_experiment_family

profile_name = sys.argv[1]
config = load_config(os.environ["CONFIG_PATH"])
profile = load_experiment_family(profile_name, config)
cluster_profile = str(profile.get("cluster_profile", "publication"))
cluster = config["cluster_registry"]["active_learning"][cluster_profile]
for key, value in cluster.items():
    print(f"cluster::{key}::{value}")
for command in profile["execution"]["audit_commands"]:
    print(f"audit_command::{command}")
PY
)

declare -A CLUSTER=()
AUDIT_COMMANDS=()
for value in "${PROFILE_VALUES[@]}"; do
  if [[ "${value}" == cluster::* ]]; then
    KEY="${value#cluster::}"
    KEY="${KEY%%::*}"
    VAL="${value##*::}"
    CLUSTER["${KEY}"]="${VAL}"
  elif [[ "${value}" == audit_command::* ]]; then
    AUDIT_COMMANDS+=("${value#audit_command::}")
  fi
done

if [[ "${#AUDIT_COMMANDS[@]}" -eq 0 ]]; then
  echo "Framework audit profile ${PROFILE_NAME} has no audit_commands."
  exit 1
fi

PROJECT_PREFIX="$(cat PROJECT_PREFIX)"
SBATCH_OUTPUT_ROOT="Output/logs"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  SBATCH_OUTPUT_ROOT="Output/logs/print_only/framework_hardening/${PROFILE_NAME}"
fi
mkdir -p "${SBATCH_OUTPUT_ROOT}"

JOB_NAME="${PROJECT_PREFIX}_audit_${PROFILE_NAME}"
SBATCH_PATH="${SBATCH_OUTPUT_ROOT}/${JOB_NAME}.sbatch"
{
  echo "#!/usr/bin/env bash"
  echo "#SBATCH --partition=${CLUSTER[partition]}"
  echo "#SBATCH --nodelist=${CLUSTER[nodelist]}"
  echo "#SBATCH --cpus-per-task=${CLUSTER[cpus_per_task]}"
  echo "#SBATCH --mem=${CLUSTER[mem]}"
  if [[ -n "${CLUSTER[gres]:-}" ]]; then
    echo "#SBATCH --gres=${CLUSTER[gres]}"
  fi
  echo "#SBATCH --job-name=${JOB_NAME}"
  echo "#SBATCH --output=Output/logs/${JOB_NAME}.%j.out"
  echo "#SBATCH --error=Output/logs/${JOB_NAME}.%j.err"
  echo
  echo "set -euo pipefail"
  echo "cd ${REPO_ROOT}"
  for command_template in "${AUDIT_COMMANDS[@]}"; do
    command="${command_template//\{config_path\}/${CONFIG_PATH}}"
    command="${command//\{python_executable\}/${PYTHON_BIN}}"
    echo "${command}"
  done
} > "${SBATCH_PATH}"
chmod +x "${SBATCH_PATH}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "Prepared ${SBATCH_PATH}"
  exit 0
fi

SBATCH_ARGS=()
if [[ -n "${UPSTREAM_DEPENDENCY_JOB_ID}" ]]; then
  SBATCH_ARGS+=("--dependency=afterok:${UPSTREAM_DEPENDENCY_JOB_ID}")
fi
AUDIT_PROFILE_JOB_ID="$(sbatch --parsable "${SBATCH_ARGS[@]}" "${SBATCH_PATH}")"
echo "AUDIT_PROFILE_JOB_ID=${AUDIT_PROFILE_JOB_ID}"
