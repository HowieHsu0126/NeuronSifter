#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-Libs/configs/config_stack.yaml}"
EXECUTION_FLAG=""
PRINT_ONLY=0
if [[ "${2:-}" == "--dry-run" ]]; then
  EXECUTION_FLAG="--dry-run"
fi
if [[ "${2:-}" == "--print-only" || "${3:-}" == "--print-only" ]]; then
  PRINT_ONLY=1
fi
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
readarray -t CLUSTER_VALUES < <(CONFIG_PATH="${CONFIG_PATH}" "${PYTHON_BIN}" - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
cluster = config["cluster_registry"]["candidate_validation"]
for scope_name in ["prep", "docking", "registry"]:
    for key, value in cluster[scope_name].items():
        print(f"{scope_name}::{key}::{value}")
PY
)

declare -A PREP_CLUSTER=()
declare -A DOCK_CLUSTER=()
declare -A REGISTRY_CLUSTER=()
for value in "${CLUSTER_VALUES[@]}"; do
  SCOPE="${value%%::*}"
  REMAINDER="${value#*::}"
  KEY="${REMAINDER%%::*}"
  VAL="${REMAINDER##*::}"
  if [[ "${SCOPE}" == "prep" ]]; then
    PREP_CLUSTER["${KEY}"]="${VAL}"
  elif [[ "${SCOPE}" == "docking" ]]; then
    DOCK_CLUSTER["${KEY}"]="${VAL}"
  else
    REGISTRY_CLUSTER["${KEY}"]="${VAL}"
  fi
done

PROJECT_PREFIX="$(cat PROJECT_PREFIX)"
PREP_JOB_NAME="${PROJECT_PREFIX}_cand_valid_prep"
PREP_SBATCH_PATH="Output/logs/${PREP_JOB_NAME}.sbatch"

{
  echo "#!/usr/bin/env bash"
  echo "#SBATCH --partition=${PREP_CLUSTER[partition]}"
  echo "#SBATCH --nodelist=${PREP_CLUSTER[nodelist]}"
  echo "#SBATCH --cpus-per-task=${PREP_CLUSTER[cpus_per_task]}"
  echo "#SBATCH --mem=${PREP_CLUSTER[mem]}"
  echo "#SBATCH --job-name=${PREP_JOB_NAME}"
  echo "#SBATCH --output=Output/logs/${PREP_JOB_NAME}.%j.out"
  echo "#SBATCH --error=Output/logs/${PREP_JOB_NAME}.%j.err"
  echo
  echo "set -euo pipefail"
  echo "cd ${REPO_ROOT}"
  echo "${PYTHON_BIN} -u Libs/exp/execution/run_candidate_validation_execution.py --config ${CONFIG_PATH} ${EXECUTION_FLAG} --execution-stage prep"
  if [[ "${EXECUTION_FLAG}" == "--dry-run" ]]; then
    echo "${PYTHON_BIN} -u Libs/scripts/builders/build_candidate_validation_registry.py --config ${CONFIG_PATH}"
    echo "${PYTHON_BIN} -u Libs/scripts/validators/validate_candidate_validation_registry.py --config ${CONFIG_PATH}"
  fi
} > "${PREP_SBATCH_PATH}"
chmod +x "${PREP_SBATCH_PATH}"

if [[ "${EXECUTION_FLAG}" == "--dry-run" ]]; then
  if [[ "${PRINT_ONLY}" -eq 1 ]]; then
    echo "Prepared ${PREP_SBATCH_PATH}"
    exit 0
  fi
  sbatch "${PREP_SBATCH_PATH}"
  exit 0
fi

DOCK_JOB_NAME="${PROJECT_PREFIX}_cand_valid_dock"
DOCK_SBATCH_PATH="Output/logs/${DOCK_JOB_NAME}.sbatch"

{
  echo "#!/usr/bin/env bash"
  echo "#SBATCH --partition=${DOCK_CLUSTER[partition]}"
  echo "#SBATCH --nodelist=${DOCK_CLUSTER[nodelist]}"
  echo "#SBATCH --cpus-per-task=${DOCK_CLUSTER[cpus_per_task]}"
  echo "#SBATCH --mem=${DOCK_CLUSTER[mem]}"
  if [[ -n "${DOCK_CLUSTER[gres]:-}" ]]; then
    echo "#SBATCH --gres=${DOCK_CLUSTER[gres]}"
  fi
  echo "#SBATCH --job-name=${DOCK_JOB_NAME}"
  echo "#SBATCH --output=Output/logs/${DOCK_JOB_NAME}.%j.out"
  echo "#SBATCH --error=Output/logs/${DOCK_JOB_NAME}.%j.err"
  echo
  echo "set -euo pipefail"
  echo "cd ${REPO_ROOT}"
  echo "${PYTHON_BIN} -u Libs/exp/execution/run_candidate_validation_execution.py --config ${CONFIG_PATH} --execution-stage docking"
} > "${DOCK_SBATCH_PATH}"
chmod +x "${DOCK_SBATCH_PATH}"

REGISTRY_JOB_NAME="${PROJECT_PREFIX}_cand_valid_registry"
REGISTRY_SBATCH_PATH="Output/logs/${REGISTRY_JOB_NAME}.sbatch"
{
  echo "#!/usr/bin/env bash"
  echo "#SBATCH --partition=${REGISTRY_CLUSTER[partition]}"
  echo "#SBATCH --nodelist=${REGISTRY_CLUSTER[nodelist]}"
  echo "#SBATCH --cpus-per-task=${REGISTRY_CLUSTER[cpus_per_task]}"
  echo "#SBATCH --mem=${REGISTRY_CLUSTER[mem]}"
  if [[ -n "${REGISTRY_CLUSTER[gres]:-}" ]]; then
    echo "#SBATCH --gres=${REGISTRY_CLUSTER[gres]}"
  fi
  echo "#SBATCH --job-name=${REGISTRY_JOB_NAME}"
  echo "#SBATCH --output=Output/logs/${REGISTRY_JOB_NAME}.%j.out"
  echo "#SBATCH --error=Output/logs/${REGISTRY_JOB_NAME}.%j.err"
  echo
  echo "set -euo pipefail"
  echo "cd ${REPO_ROOT}"
  echo "${PYTHON_BIN} -u Libs/scripts/builders/build_candidate_validation_registry.py --config ${CONFIG_PATH}"
  echo "${PYTHON_BIN} -u Libs/scripts/validators/validate_candidate_validation_registry.py --config ${CONFIG_PATH}"
} > "${REGISTRY_SBATCH_PATH}"
chmod +x "${REGISTRY_SBATCH_PATH}"

if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  echo "Prepared ${PREP_SBATCH_PATH}"
  echo "Prepared ${DOCK_SBATCH_PATH}"
  echo "Prepared ${REGISTRY_SBATCH_PATH}"
  exit 0
fi

"${PYTHON_BIN}" - "$CONFIG_PATH" <<'PY'
import os
import sys

from Libs.utils.data.raw_data_registry import load_config

config = load_config(sys.argv[1])
binaries = config["validation"]["candidate_validation_execution"]["binaries"]
for name in ["vina", "gnina", "obabel"]:
    path = str(binaries[name])
    if not os.path.isfile(path) or not os.access(path, os.X_OK):
        raise SystemExit(f"Missing executable candidate-validation binary: {path}")
PY

PREP_JOB_ID="$(sbatch --parsable "${PREP_SBATCH_PATH}")"
DOCK_JOB_ID="$(sbatch --dependency=afterok:${PREP_JOB_ID} --parsable "${DOCK_SBATCH_PATH}")"
sbatch --dependency=afterok:${DOCK_JOB_ID} "${REGISTRY_SBATCH_PATH}"
