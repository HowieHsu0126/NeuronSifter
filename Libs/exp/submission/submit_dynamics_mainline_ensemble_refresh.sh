#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--print-only" ]]; then
  DRY_RUN=1
fi
CONFIG_PATH="${2:-Libs/configs/config_stack.yaml}"

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
readarray -t ENSEMBLE_VALUES < <(CONFIG_PATH="${CONFIG_PATH}" "${PYTHON_BIN}" - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
cluster = config["cluster_registry"]["dynamics"]["ensemble_refresh"]
for key, value in cluster.items():
    if key == "seed_values":
        for seed_value in value:
            print(f"seed::{seed_value}")
        continue
    print(f"cluster::{key}::{value}")
PY
)

PROJECT_PREFIX="$(cat PROJECT_PREFIX)"
SEED_VALUES=()
declare -A ENSEMBLE_CLUSTER=()
for value in "${ENSEMBLE_VALUES[@]}"; do
  if [[ "${value}" == seed::* ]]; then
    SEED_VALUES+=("${value#seed::}")
    continue
  fi
  KEY="${value#cluster::}"
  KEY="${KEY%%::*}"
  VAL="${value##*::}"
  ENSEMBLE_CLUSTER["${KEY}"]="${VAL}"
done

for SEED_VALUE in "${SEED_VALUES[@]}"; do
  OUTPUT_CONFIG="Output/paper_configs/dynamics_mainline_ad_seed_${SEED_VALUE}.yaml"
  "${PYTHON_BIN}" Libs/exp/orchestration/prepare_experiment_config.py \
    --profile dynamics_mainline_ad \
    --seed "${SEED_VALUE}" \
    --output-config "${OUTPUT_CONFIG}" \
    --base-config "${CONFIG_PATH}"

  JOB_NAME="${PROJECT_PREFIX}_mainline_eval_${SEED_VALUE}"
  SBATCH_PATH="Output/logs/${JOB_NAME}.sbatch"
  {
    echo "#!/usr/bin/env bash"
    echo "#SBATCH --partition=${ENSEMBLE_CLUSTER[partition]}"
    echo "#SBATCH --nodelist=${ENSEMBLE_CLUSTER[nodelist]}"
    echo "#SBATCH --cpus-per-task=${ENSEMBLE_CLUSTER[cpus_per_task]}"
    echo "#SBATCH --mem=${ENSEMBLE_CLUSTER[mem]}"
    if [[ -n "${ENSEMBLE_CLUSTER[gres]:-}" ]]; then
      echo "#SBATCH --gres=${ENSEMBLE_CLUSTER[gres]}"
    fi
    echo "#SBATCH --job-name=${JOB_NAME}"
    echo "#SBATCH --output=Output/logs/${JOB_NAME}.%j.out"
    echo "#SBATCH --error=Output/logs/${JOB_NAME}.%j.err"
    echo
    echo "set -euo pipefail"
    echo "cd ${REPO_ROOT}"
    echo "${PYTHON_BIN} -u Libs/exp/evaluation/evaluate_screening_system.py --config ${OUTPUT_CONFIG} --ensemble-mode required"
  } > "${SBATCH_PATH}"
  chmod +x "${SBATCH_PATH}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Prepared ${SBATCH_PATH}"
    continue
  fi

  sbatch "${SBATCH_PATH}"
done
