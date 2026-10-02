#!/usr/bin/env bash
set -euo pipefail

PRINT_ONLY=0
CONFIG_PATH="Libs/configs/config_stack.yaml"
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
cd "${REPO_ROOT}"

sinfo

PYTHON_BIN="$(CONFIG_PATH="${CONFIG_PATH}" python - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
print(str(config["execution_environment"]["python_executable"]))
PY
)"

readarray -t FSDP_VALUES < <(CONFIG_PATH="${CONFIG_PATH}" "${PYTHON_BIN}" - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
registry = config["neuronsifter_scale_registry"]
checkpoint_scale = str(registry["default_fsdp_smoke_checkpoint_scale"])
inference_profile = str(registry["default_fsdp_smoke_inference_profile"])
scale_config = registry["checkpoint_scales"][checkpoint_scale]
profile = registry["fsdp_resource_profiles"][scale_config["fsdp_resource_profile"]]
print(f"checkpoint_scale::{checkpoint_scale}")
print(f"inference_profile::{inference_profile}")
for key, value in profile.items():
    print(f"{key}::{value}")
PY
)

fsdp_value() {
  local key_name="$1"
  local row
  for row in "${FSDP_VALUES[@]}"; do
    local row_key="${row%%::*}"
    local row_value="${row#*::}"
    if [[ "${row_key}" == "${key_name}" ]]; then
      printf '%s\n' "${row_value}"
      return 0
    fi
  done
  return 1
}

PROJECT_PREFIX="$(cat PROJECT_PREFIX)"
JOB_SUFFIX="$(fsdp_value job_suffix)"
JOB_NAME="${PROJECT_PREFIX}_${JOB_SUFFIX}"
SBATCH_OUTPUT_ROOT="Output/logs"
if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  SBATCH_OUTPUT_ROOT="Output/logs/print_only/fsdp"
fi
mkdir -p "${SBATCH_OUTPUT_ROOT}"
SBATCH_PATH="${SBATCH_OUTPUT_ROOT}/${JOB_NAME}.sbatch"

{
  echo "#!/usr/bin/env bash"
  echo "#SBATCH --partition=$(fsdp_value partition)"
  echo "#SBATCH --nodelist=$(fsdp_value nodelist)"
  echo "#SBATCH --gres=gpu:$(fsdp_value gpu_count)"
  echo "#SBATCH --cpus-per-task=$(fsdp_value cpus_per_task)"
  echo "#SBATCH --mem=$(fsdp_value mem)"
  echo "#SBATCH --job-name=${JOB_NAME}"
  echo "#SBATCH --output=Output/logs/${JOB_NAME}.%j.out"
  echo "#SBATCH --error=Output/logs/${JOB_NAME}.%j.err"
  echo
  echo "set -euo pipefail"
  echo "cd ${REPO_ROOT}"
  echo "srun ${PYTHON_BIN} -m torch.distributed.run --standalone --nproc_per_node $(fsdp_value gpu_count) Libs/exp/training/pretrain_microenvironment_world_model.py --config ${CONFIG_PATH} --checkpoint-scale $(fsdp_value checkpoint_scale) --inference-profile $(fsdp_value inference_profile) --distributed-strategy $(fsdp_value distributed_strategy) --device cuda"
} > "${SBATCH_PATH}"
chmod +x "${SBATCH_PATH}"

if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  echo "Prepared ${SBATCH_PATH}"
  exit 0
fi

SBATCH_ARGS=()
if [[ -n "${UPSTREAM_DEPENDENCY_JOB_ID}" ]]; then
  SBATCH_ARGS+=("--dependency=afterok:${UPSTREAM_DEPENDENCY_JOB_ID}")
fi
FSDP_JOB_ID="$(sbatch --parsable "${SBATCH_ARGS[@]}" "${SBATCH_PATH}")"
echo "FSDP_JOB_ID=${FSDP_JOB_ID}"
