#!/usr/bin/env bash
set -euo pipefail

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

readarray -t SUITE_VALUES < <(CONFIG_PATH="${CONFIG_PATH}" "${PYTHON_BIN}" - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
cluster = config["baseline_adapter_registry"]["cluster"]["suite_submission"]
suite = config["baseline_adapter_registry"]["default_suite"]
paper_suite = config.get("paper_grade_full_suite", {})
for key, value in cluster.items():
    print(f"cluster::{key}::{value}")
print(f"suite::disease_program::{suite['disease_program']}")
print(f"suite::skip_existing::{suite.get('skip_existing', False)}")
print(f"freeze::sha256::{paper_suite.get('freeze_snapshot_signature_sha256', '')}")
for baseline_name in suite["baseline_names"]:
    print(f"baseline::{baseline_name}")
for seed_value in paper_suite.get("formal_seed_values", []):
    print(f"seed::{int(seed_value)}")
PY
)

declare -A CLUSTER=()
BASELINE_NAMES=()
SEED_VALUES=()
DISEASE_PROGRAM=""
FREEZE_SIGNATURE_SHA256=""
SKIP_EXISTING=0
for value in "${SUITE_VALUES[@]}"; do
  if [[ "${value}" == cluster::* ]]; then
    KEY="${value#cluster::}"
    KEY="${KEY%%::*}"
    VAL="${value##*::}"
    CLUSTER["${KEY}"]="${VAL}"
  elif [[ "${value}" == suite::disease_program::* ]]; then
    DISEASE_PROGRAM="${value##*::}"
  elif [[ "${value}" == suite::skip_existing::* ]]; then
    if [[ "${value##*::}" == "True" || "${value##*::}" == "true" ]]; then
      SKIP_EXISTING=1
    fi
  elif [[ "${value}" == freeze::sha256::* ]]; then
    FREEZE_SIGNATURE_SHA256="${value#freeze::sha256::}"
  elif [[ "${value}" == baseline::* ]]; then
    BASELINE_NAMES+=("${value#baseline::}")
  elif [[ "${value}" == seed::* ]]; then
    SEED_VALUES+=("${value#seed::}")
  fi
done

if [[ "${#SEED_VALUES[@]}" -eq 0 ]]; then
  SEED_VALUES=("")
fi

PROJECT_PREFIX="$(cat PROJECT_PREFIX)"
SBATCH_OUTPUT_ROOT="Output/logs"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  SBATCH_OUTPUT_ROOT="Output/logs/print_only/baselines"
fi
mkdir -p "${SBATCH_OUTPUT_ROOT}"
SUBMITTED_JOB_IDS=()

for BASELINE_NAME in "${BASELINE_NAMES[@]}"; do
  for SEED_VALUE in "${SEED_VALUES[@]}"; do
  SEED_SUFFIX=""
  if [[ -n "${SEED_VALUE}" ]]; then
    SEED_SUFFIX="_seed_${SEED_VALUE}"
  fi
  JOB_NAME="${PROJECT_PREFIX}_baseline_${BASELINE_NAME}${SEED_SUFFIX}"
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
    if [[ -n "${FREEZE_SIGNATURE_SHA256}" ]]; then
      echo "export PAPER_GRADE_FREEZE_SIGNATURE_SHA256=${FREEZE_SIGNATURE_SHA256}"
    fi
    echo "cd ${REPO_ROOT}"
    COMMAND="${PYTHON_BIN} -u Libs/exp/baselines/run_runnable_baseline_suite.py --config ${CONFIG_PATH} --baseline-name ${BASELINE_NAME} --disease-program ${DISEASE_PROGRAM}"
    if [[ -n "${SEED_VALUE}" ]]; then
      COMMAND="${COMMAND} --seed ${SEED_VALUE}"
    fi
    if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
      COMMAND="${COMMAND} --skip-existing"
    fi
    echo "${COMMAND}"
  } > "${SBATCH_PATH}"
  chmod +x "${SBATCH_PATH}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Prepared ${SBATCH_PATH}"
    continue
  fi

  SBATCH_ARGS=()
  if [[ -n "${UPSTREAM_DEPENDENCY_JOB_ID}" ]]; then
    SBATCH_ARGS+=("--dependency=afterok:${UPSTREAM_DEPENDENCY_JOB_ID}")
  fi
  SUBMITTED_JOB_IDS+=("$(sbatch --parsable "${SBATCH_ARGS[@]}" "${SBATCH_PATH}")")
  done
done

if [[ "${DRY_RUN}" -eq 0 && "${#SUBMITTED_JOB_IDS[@]}" -gt 0 ]]; then
  BASELINE_DEPENDENCY_VALUE="$(IFS=:; echo "${SUBMITTED_JOB_IDS[*]}")"
  echo "BASELINE_JOB_IDS=${BASELINE_DEPENDENCY_VALUE}"
fi
