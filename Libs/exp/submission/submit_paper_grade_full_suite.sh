#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="Libs/configs/config_stack.yaml"
PRINT_ONLY=0
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

PROJECT_PREFIX="$(cat PROJECT_PREFIX)"
if [[ "${PROJECT_PREFIX}" != "NrnSift" ]]; then
  echo "PROJECT_PREFIX must be NrnSift for NeuronSifter paper-grade suite submissions."
  exit 1
fi
if [[ "${PRINT_ONLY}" -eq 0 ]]; then
  EXISTING_PREFIX_JOBS="$(squeue -h -u "${USER}" -o "%i %j" | awk -v prefix="${PROJECT_PREFIX}_" 'index($2, prefix) == 1 {print}' || true)"
  if [[ -n "${EXISTING_PREFIX_JOBS}" ]]; then
    echo "Refusing live submission because exact-prefix NrnSift_ jobs are still visible:"
    printf '%s\n' "${EXISTING_PREFIX_JOBS}"
    exit 1
  fi
fi

PYTHON_BIN="$(CONFIG_PATH="${CONFIG_PATH}" python - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
print(str(config["execution_environment"]["python_executable"]))
PY
)"

MATERIALIZED_CONFIG="$(CONFIG_PATH="${CONFIG_PATH}" "${PYTHON_BIN}" - <<'PY'
import os
import hashlib
import json
from pathlib import Path

import yaml

from Libs.utils.data.raw_data_registry import ensure_directory
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.raw_data_registry import write_text_atomic

config_path = os.environ["CONFIG_PATH"]
config = load_config(config_path)
suite = config["paper_grade_full_suite"]
output_path = Path(str(suite["cluster"]["suite_config_output"]))
freeze_manifest_path = Path("Output/audits/formal_experiment_freeze_manifest.json")
freeze_manifest = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
freeze_signature = freeze_manifest["snapshot_signature"]
freeze_signature_sha256 = hashlib.sha256(
    json.dumps(freeze_signature, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
).hexdigest()
ensure_directory(output_path.parent)
payload = {
    "config_includes": [config_path],
    "paper_grade_full_suite": {
        "suite_config_materialized": True,
        "freeze_replay_config_path": config_path,
        "freeze_manifest_path": freeze_manifest_path.as_posix(),
        "freeze_snapshot_signature": freeze_signature,
        "freeze_snapshot_signature_sha256": freeze_signature_sha256,
    },
    "screening_pipeline": {
        "screening_benchmark": {
            "formal_seed_values": [int(value) for value in suite["formal_seed_values"]],
        },
    },
    "baseline_adapter_registry": {
        "default_suite": {
            "disease_program": "ad_control",
            "skip_existing": False,
            "baseline_names": [str(value) for value in suite["submission_order"]["baseline_adapters"]],
        },
    },
}
write_text_atomic(output_path, yaml.safe_dump(payload, sort_keys=False))
print(output_path.as_posix())
PY
)"

readarray -t SUITE_VALUES < <(CONFIG_PATH="${MATERIALIZED_CONFIG}" "${PYTHON_BIN}" - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
suite = config["paper_grade_full_suite"]
for command in suite["validators"]["prelaunch_order"]:
    print(f"gate::{command}")
for profile_name in suite["submission_order"]["dynamics_profiles"]:
    print(f"dynamics::{profile_name}")
for profile_name in suite["submission_order"].get("shared_context_profiles", []):
    print(f"shared::{profile_name}")
for key, value in suite["cluster"]["cpu_gate"].items():
    print(f"cpu_gate::{key}::{value}")
for key, value in suite["cluster"]["cpu_large"].items():
    print(f"cpu_large::{key}::{value}")
print(f"paths::print_only_root::{suite['cluster']['print_only_root']}")
print(f"freeze::sha256::{suite.get('freeze_snapshot_signature_sha256', '')}")
PY
)

GATE_COMMANDS=()
DYNAMICS_PROFILES=()
SHARED_CONTEXT_PROFILES=()
FREEZE_SIGNATURE_SHA256=""
declare -A CPU_GATE=()
declare -A CPU_LARGE=()
PRINT_ONLY_ROOT="Output/logs/print_only/paper_grade_full_suite"
for value in "${SUITE_VALUES[@]}"; do
  if [[ "${value}" == gate::* ]]; then
    GATE_COMMANDS+=("${value#gate::}")
  elif [[ "${value}" == dynamics::* ]]; then
    DYNAMICS_PROFILES+=("${value#dynamics::}")
  elif [[ "${value}" == shared::* ]]; then
    SHARED_CONTEXT_PROFILES+=("${value#shared::}")
  elif [[ "${value}" == cpu_gate::* ]]; then
    KEY="${value#cpu_gate::}"
    KEY="${KEY%%::*}"
    VAL="${value#cpu_gate::${KEY}::}"
    CPU_GATE["${KEY}"]="${VAL}"
  elif [[ "${value}" == cpu_large::* ]]; then
    KEY="${value#cpu_large::}"
    KEY="${KEY%%::*}"
    VAL="${value#cpu_large::${KEY}::}"
    CPU_LARGE["${KEY}"]="${VAL}"
  elif [[ "${value}" == paths::print_only_root::* ]]; then
    PRINT_ONLY_ROOT="${value#paths::print_only_root::}"
  elif [[ "${value}" == freeze::sha256::* ]]; then
    FREEZE_SIGNATURE_SHA256="${value#freeze::sha256::}"
  fi
done

if [[ "${#DYNAMICS_PROFILES[@]}" -eq 0 ]]; then
  echo "paper_grade_full_suite.submission_order.dynamics_profiles is empty."
  exit 1
fi

remove_print_only_path() {
  local path="$1"
  case "${path}" in
    Output/logs/print_only/*|Output/paper_configs/print_only/*)
      rm -rf "${path}"
      ;;
    *)
      echo "Refusing to remove non-print-only path: ${path}"
      exit 1
      ;;
  esac
}

if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  for profile_name in "${DYNAMICS_PROFILES[@]}" "${SHARED_CONTEXT_PROFILES[@]}"; do
    remove_print_only_path "Output/logs/print_only/${profile_name}"
    remove_print_only_path "Output/paper_configs/print_only/${profile_name}"
  done
  remove_print_only_path "Output/logs/print_only/baselines"
  remove_print_only_path "${PRINT_ONLY_ROOT}"
fi

if [[ "${PRINT_ONLY}" -eq 0 ]]; then
  for gate_command in "${GATE_COMMANDS[@]}"; do
    read -r -a gate_parts <<< "${gate_command}"
    "${PYTHON_BIN}" -u "${gate_parts[@]}" --config "${MATERIALIZED_CONFIG}"
  done
fi

PRINT_ONLY_ARG=()
SBATCH_OUTPUT_ROOT="Output/logs/paper_grade_full_suite"
if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  PRINT_ONLY_ARG+=("--print-only")
  SBATCH_OUTPUT_ROOT="${PRINT_ONLY_ROOT}"
fi
mkdir -p "${SBATCH_OUTPUT_ROOT}"

LAST_DYNAMICS_JOB_ID=""
DEPENDENCY_ARG=()
for profile_name in "${DYNAMICS_PROFILES[@]}"; do
  output="$(bash "${SCRIPT_DIR}/submit_dynamics_profile_suite.sh" "${profile_name}" --config "${MATERIALIZED_CONFIG}" "${PRINT_ONLY_ARG[@]}" "${DEPENDENCY_ARG[@]}")"
  printf '%s\n' "${output}"
  if [[ "${PRINT_ONLY}" -eq 0 ]]; then
    LAST_DYNAMICS_JOB_ID="$(printf '%s\n' "${output}" | awk -F= '$1 == "PROFILE_LAST_JOB_ID" {print $2}' | tail -n 1)"
    if [[ -z "${LAST_DYNAMICS_JOB_ID}" ]]; then
      echo "Failed to parse PROFILE_LAST_JOB_ID for ${profile_name}."
      exit 1
    fi
    DEPENDENCY_ARG=("--dependency-job-id" "${LAST_DYNAMICS_JOB_ID}")
  fi
done

SHARED_JOB_IDS=()
for profile_name in "${SHARED_CONTEXT_PROFILES[@]}"; do
  SHARED_ARGS=("--config" "${MATERIALIZED_CONFIG}" "${PRINT_ONLY_ARG[@]}")
  if [[ "${PRINT_ONLY}" -eq 0 && -n "${LAST_DYNAMICS_JOB_ID}" ]]; then
    SHARED_ARGS+=("--dependency-job-id" "${LAST_DYNAMICS_JOB_ID}")
  fi
  output="$(bash "${SCRIPT_DIR}/submit_profile_suite.sh" "${profile_name}" "${SHARED_ARGS[@]}")"
  printf '%s\n' "${output}"
  if [[ "${PRINT_ONLY}" -eq 0 ]]; then
    SHARED_LAST_JOB_ID="$(printf '%s\n' "${output}" | awk -F= '$1 == "PROFILE_LAST_JOB_ID" {print $2}' | tail -n 1)"
    if [[ -z "${SHARED_LAST_JOB_ID}" ]]; then
      echo "Failed to parse PROFILE_LAST_JOB_ID for ${profile_name}."
      exit 1
    fi
    SHARED_JOB_IDS+=("${SHARED_LAST_JOB_ID}")
  fi
done

BASELINE_JOB_IDS=""
BASELINE_ARGS=("--config" "${MATERIALIZED_CONFIG}" "${PRINT_ONLY_ARG[@]}")
if [[ "${PRINT_ONLY}" -eq 0 && -n "${LAST_DYNAMICS_JOB_ID}" ]]; then
  BASELINE_ARGS+=("--dependency-job-id" "${LAST_DYNAMICS_JOB_ID}")
fi
BASELINE_OUTPUT="$(bash "${SCRIPT_DIR}/submit_runnable_baseline_suite.sh" "${BASELINE_ARGS[@]}")"
printf '%s\n' "${BASELINE_OUTPUT}"
if [[ "${PRINT_ONLY}" -eq 0 ]]; then
  BASELINE_JOB_IDS="$(printf '%s\n' "${BASELINE_OUTPUT}" | awk -F= '$1 == "BASELINE_JOB_IDS" {print $2}' | tail -n 1)"
fi

write_cpu_sbatch() {
  local job_name="$1"
  local sbatch_path="$2"
  local command="$3"
  {
    echo "#!/usr/bin/env bash"
    echo "#SBATCH --partition=${CPU_GATE[partition]}"
    echo "#SBATCH --nodelist=${CPU_GATE[nodelist]}"
    echo "#SBATCH --cpus-per-task=${CPU_GATE[cpus_per_task]}"
    echo "#SBATCH --mem=${CPU_GATE[mem]}"
    if [[ -n "${CPU_GATE[gres]:-}" ]]; then
      echo "#SBATCH --gres=${CPU_GATE[gres]}"
    fi
    echo "#SBATCH --job-name=${job_name}"
    echo "#SBATCH --output=Output/logs/${job_name}.%j.out"
    echo "#SBATCH --error=Output/logs/${job_name}.%j.err"
    echo
    echo "set -euo pipefail"
    if [[ -n "${FREEZE_SIGNATURE_SHA256}" ]]; then
      echo "export PAPER_GRADE_FREEZE_SIGNATURE_SHA256=${FREEZE_SIGNATURE_SHA256}"
    fi
    echo "cd ${REPO_ROOT}"
    echo "${command}"
  } > "${sbatch_path}"
  chmod +x "${sbatch_path}"
}

EVAL_JOB_NAME="${PROJECT_PREFIX}_paper_grade_ad_reproduction"
EVAL_SBATCH="${SBATCH_OUTPUT_ROOT}/${EVAL_JOB_NAME}.sbatch"
write_cpu_sbatch \
  "${EVAL_JOB_NAME}" \
  "${EVAL_SBATCH}" \
  "${PYTHON_BIN} -u Libs/exp/evaluation/evaluate_published_ad_sift_reproduction.py --config ${MATERIALIZED_CONFIG}"

CLOSEOUT_JOB_NAME="${PROJECT_PREFIX}_paper_grade_ad_closeout"
CLOSEOUT_SBATCH="${SBATCH_OUTPUT_ROOT}/${CLOSEOUT_JOB_NAME}.sbatch"
write_cpu_sbatch \
  "${CLOSEOUT_JOB_NAME}" \
  "${CLOSEOUT_SBATCH}" \
  "${PYTHON_BIN} -u Libs/exp/evaluation/evaluate_published_ad_sift_reproduction.py --config ${MATERIALIZED_CONFIG}
${PYTHON_BIN} -u Libs/scripts/validators/validate_paper_grade_full_suite.py --config ${MATERIALIZED_CONFIG} --mode postrun"

if [[ "${PRINT_ONLY}" -eq 1 ]]; then
  echo "Prepared ${EVAL_SBATCH}"
  echo "Prepared ${CLOSEOUT_SBATCH}"
  exit 0
fi

EVAL_DEPENDENCY_ARGS=()
if [[ -n "${LAST_DYNAMICS_JOB_ID}" ]]; then
  EVAL_DEPENDENCY_ARGS+=("--dependency=afterok:${LAST_DYNAMICS_JOB_ID}")
fi
EVAL_JOB_ID="$(sbatch --parsable "${EVAL_DEPENDENCY_ARGS[@]}" "${EVAL_SBATCH}")"

CLOSEOUT_DEPENDENCIES=("${EVAL_JOB_ID}")
if [[ -n "${LAST_DYNAMICS_JOB_ID}" ]]; then
  CLOSEOUT_DEPENDENCIES+=("${LAST_DYNAMICS_JOB_ID}")
fi
if [[ "${#SHARED_JOB_IDS[@]}" -gt 0 ]]; then
  CLOSEOUT_DEPENDENCIES+=("${SHARED_JOB_IDS[@]}")
fi
if [[ -n "${BASELINE_JOB_IDS}" ]]; then
  CLOSEOUT_DEPENDENCIES+=("${BASELINE_JOB_IDS}")
fi
CLOSEOUT_DEPENDENCY_VALUE="$(IFS=:; echo "${CLOSEOUT_DEPENDENCIES[*]}")"
CLOSEOUT_JOB_ID="$(sbatch --parsable --dependency=afterok:${CLOSEOUT_DEPENDENCY_VALUE} "${CLOSEOUT_SBATCH}")"

echo "PAPER_GRADE_EVAL_JOB_ID=${EVAL_JOB_ID}"
echo "PAPER_GRADE_CLOSEOUT_JOB_ID=${CLOSEOUT_JOB_ID}"
