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

from Libs.utils.exp.experiment_manifests import deep_merge, load_experiment_family
from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path

profile_name = sys.argv[1]
config = load_config(os.environ["CONFIG_PATH"])
profile = load_experiment_family(profile_name, config)
base_config = load_config(profile["base_config_path"])
materialized_config = deep_merge(base_config, profile.get("config_overrides", {}))
runtime = profile.get("runtime", {})
cluster_defaults = config.get("cluster_registry", {}).get("dynamics", {}).get("profile_defaults", {})
prebuild = deep_merge(cluster_defaults.get("prebuild", {}), profile.get("cluster_prebuild", {}))
training = deep_merge(cluster_defaults.get("training", {}), profile.get("cluster_training", profile.get("cluster", {})))
evaluation = dict(config["paper_grade_full_suite"]["cluster"]["cpu_large"])
execution = profile.get("execution", {})
readiness = materialized_config["dynamics_ready_registry"]["ad_first_data_readiness"]
suite = dict(config["paper_grade_full_suite"])
runtime_protocol = dict(suite.get("runtime_protocol", {}))
screening_training = dict(runtime_protocol.get("screening_training", {}))
split_commands = dict(runtime_protocol.get("commands", {}))
cpu_precompute = dict(runtime_protocol.get("cpu_precompute", {}))
disease_program = str(profile.get("disease_program", runtime.get("disease_program", "ad_control")))
fold_catalog = read_tsv(resolve_repo_path(config["benchmark_registry"]["local_root"]) / config["benchmark_registry"]["relative_paths"]["fold_catalog"])

def emit(prefix, payload):
    for key, value in payload.items():
        print(f"{prefix}::{key}::{value}")

emit("prebuild", prebuild)
emit("training", training)
emit("evaluation", evaluation)
print(f"runtime::device::{runtime.get('device', 'cpu')}")
print(f"runtime::training_scope::{runtime.get('training_scope', 'all')}")
print(f"runtime::disease_program::{disease_program}")
print(f"runtime::generation_stage::{readiness['active_generation_stage']}")
print(f"runtime::primary_disease_program::{readiness['primary_disease_program']}")
print(f"freeze::sha256::{config.get('paper_grade_full_suite', {}).get('freeze_snapshot_signature_sha256', '')}")
print(f"protocol::checkpointable_split_enabled::{bool(runtime_protocol.get('checkpointable_split_enabled', False))}")
print(f"protocol::submission_throttle_seconds::{runtime_protocol.get('submission_throttle_seconds', 0)}")
print(f"protocol::max_epochs::{screening_training.get('max_epochs', '')}")
print(f"protocol::patience_epochs::{screening_training.get('patience_epochs', '')}")
print(f"protocol::epoch_window_count::{screening_training.get('epoch_window_count', '')}")
print(f"protocol::cpu_precompute_enabled::{bool(cpu_precompute.get('enabled', False))}")
for label in screening_training.get("epoch_window_labels", []):
    print(f"window_label::{label}")
for key, command in split_commands.items():
    print(f"split_command::{key}::{command}")
for row in fold_catalog:
    if row.get("disease_program", disease_program) == disease_program:
        print(f"fold::{row['fold_name']}")
for seed_value in runtime.get("seed_values", []):
    print(f"seed::{seed_value}")
for command in execution.get("prebuild_commands", []):
    print(f"prebuild_command::{command}")
for command in execution.get("train_commands", []):
    print(f"train_command::{command}")
PY
)

PROJECT_PREFIX="$(cat PROJECT_PREFIX)"
mkdir -p Output/locks
CONFIG_OUTPUT_ROOT="Output/paper_configs"
SBATCH_OUTPUT_ROOT="Output/logs"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  CONFIG_OUTPUT_ROOT="Output/paper_configs/print_only/${PROFILE_NAME}"
  SBATCH_OUTPUT_ROOT="Output/logs/print_only/${PROFILE_NAME}"
fi
mkdir -p "${CONFIG_OUTPUT_ROOT}" "${SBATCH_OUTPUT_ROOT}"
declare -A PREBUILD_CLUSTER=()
declare -A TRAINING_CLUSTER=()
declare -A EVALUATION_CLUSTER=()
DEVICE_NAME="cpu"
TRAINING_SCOPE="all"
DISEASE_PROGRAM=""
GENERATION_STAGE="data"
PRIMARY_DISEASE_PROGRAM="ad_control"
FREEZE_SIGNATURE_SHA256=""
CHECKPOINTABLE_SPLIT_ENABLED="False"
SUBMISSION_THROTTLE_SECONDS="0"
SCREENING_MAX_EPOCHS=""
SCREENING_PATIENCE_EPOCHS=""
SCREENING_EPOCH_WINDOW_COUNT=""
CPU_PRECOMPUTE_ENABLED="False"
SEED_VALUES=()
FOLD_NAMES=()
WINDOW_LABELS=()
declare -A SPLIT_COMMANDS=()
PREBUILD_COMMANDS=()
TRAIN_COMMANDS=()
for value in "${PROFILE_VALUES[@]}"; do
  if [[ "${value}" == prebuild::* ]]; then
    KEY="${value#prebuild::}"
    KEY="${KEY%%::*}"
    VAL="${value#prebuild::${KEY}::}"
    PREBUILD_CLUSTER["${KEY}"]="${VAL}"
  elif [[ "${value}" == training::* ]]; then
    KEY="${value#training::}"
    KEY="${KEY%%::*}"
    VAL="${value#training::${KEY}::}"
    TRAINING_CLUSTER["${KEY}"]="${VAL}"
  elif [[ "${value}" == evaluation::* ]]; then
    KEY="${value#evaluation::}"
    KEY="${KEY%%::*}"
    VAL="${value#evaluation::${KEY}::}"
    EVALUATION_CLUSTER["${KEY}"]="${VAL}"
  elif [[ "${value}" == runtime::device::* ]]; then
    DEVICE_NAME="${value##*::}"
  elif [[ "${value}" == runtime::training_scope::* ]]; then
    TRAINING_SCOPE="${value##*::}"
  elif [[ "${value}" == runtime::disease_program::* ]]; then
    DISEASE_PROGRAM="${value##*::}"
  elif [[ "${value}" == runtime::generation_stage::* ]]; then
    GENERATION_STAGE="${value##*::}"
  elif [[ "${value}" == runtime::primary_disease_program::* ]]; then
    PRIMARY_DISEASE_PROGRAM="${value##*::}"
  elif [[ "${value}" == freeze::sha256::* ]]; then
    FREEZE_SIGNATURE_SHA256="${value#freeze::sha256::}"
  elif [[ "${value}" == protocol::checkpointable_split_enabled::* ]]; then
    CHECKPOINTABLE_SPLIT_ENABLED="${value#protocol::checkpointable_split_enabled::}"
  elif [[ "${value}" == protocol::submission_throttle_seconds::* ]]; then
    SUBMISSION_THROTTLE_SECONDS="${value#protocol::submission_throttle_seconds::}"
  elif [[ "${value}" == protocol::max_epochs::* ]]; then
    SCREENING_MAX_EPOCHS="${value#protocol::max_epochs::}"
  elif [[ "${value}" == protocol::patience_epochs::* ]]; then
    SCREENING_PATIENCE_EPOCHS="${value#protocol::patience_epochs::}"
  elif [[ "${value}" == protocol::epoch_window_count::* ]]; then
    SCREENING_EPOCH_WINDOW_COUNT="${value#protocol::epoch_window_count::}"
  elif [[ "${value}" == protocol::cpu_precompute_enabled::* ]]; then
    CPU_PRECOMPUTE_ENABLED="${value#protocol::cpu_precompute_enabled::}"
  elif [[ "${value}" == window_label::* ]]; then
    WINDOW_LABELS+=("${value#window_label::}")
  elif [[ "${value}" == split_command::* ]]; then
    KEY="${value#split_command::}"
    KEY="${KEY%%::*}"
    VAL="${value#split_command::${KEY}::}"
    SPLIT_COMMANDS["${KEY}"]="${VAL}"
  elif [[ "${value}" == fold::* ]]; then
    FOLD_NAMES+=("${value#fold::}")
  elif [[ "${value}" == seed::* ]]; then
    SEED_VALUES+=("${value#seed::}")
  elif [[ "${value}" == prebuild_command::* ]]; then
    PREBUILD_COMMANDS+=("${value#prebuild_command::}")
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

if [[ "${CHECKPOINTABLE_SPLIT_ENABLED}" == "True" ]]; then
  REQUIRED_COMMAND_KEYS=(pretraining screening_window evaluation active_learning_replay export)
  if [[ "${CPU_PRECOMPUTE_ENABLED}" == "True" ]]; then
    REQUIRED_COMMAND_KEYS=(cache_precompute "${REQUIRED_COMMAND_KEYS[@]}")
  fi
  for command_key in "${REQUIRED_COMMAND_KEYS[@]}"; do
    if [[ -z "${SPLIT_COMMANDS[${command_key}]:-}" ]]; then
      echo "paper_grade_full_suite.runtime_protocol.commands.${command_key} is required when checkpointable split is enabled."
      exit 1
    fi
  done
  if [[ -z "${SCREENING_MAX_EPOCHS}" || -z "${SCREENING_PATIENCE_EPOCHS}" || -z "${SCREENING_EPOCH_WINDOW_COUNT}" ]]; then
    echo "paper_grade_full_suite.runtime_protocol.screening_training must declare max_epochs, patience_epochs, and epoch_window_count."
    exit 1
  fi
  if [[ "${#WINDOW_LABELS[@]}" -ne "${SCREENING_EPOCH_WINDOW_COUNT}" ]]; then
    echo "paper_grade_full_suite.runtime_protocol.screening_training.epoch_window_labels must match epoch_window_count."
    exit 1
  fi
  if [[ "${#FOLD_NAMES[@]}" -eq 0 ]]; then
    echo "No fold names were found for ${DISEASE_PROGRAM}."
    exit 1
  fi
fi

if [[ "${DEVICE_NAME}" == cuda* && -z "${TRAINING_CLUSTER[pytorch_alloc_conf]:-}" ]]; then
  echo "CUDA dynamics profile ${PROFILE_NAME} must declare cluster_registry.dynamics.profile_defaults.training.pytorch_alloc_conf."
  exit 1
fi

TRAIN_CHAIN_STATE_PATH="Output/locks/dynamics_gpu_train_chain_last_job.txt"
PROFILE_CHAIN_STATE_PATH="Output/locks/dynamics_profile_${PROFILE_NAME}_last_job.txt"
LAST_TRAIN_JOB_ID=""
if [[ "${DRY_RUN}" -eq 0 && -f "${TRAIN_CHAIN_STATE_PATH}" ]]; then
  CANDIDATE_TRAIN_JOB_ID="$(cat "${TRAIN_CHAIN_STATE_PATH}")"
  ACTIVE_TRAIN_JOB_ID=""
  if [[ -n "${CANDIDATE_TRAIN_JOB_ID}" ]]; then
    ACTIVE_TRAIN_JOB_ID="$(squeue -h -j "${CANDIDATE_TRAIN_JOB_ID}" -o "%i" 2>/dev/null | head -n 1 || true)"
  fi
  if [[ -n "${CANDIDATE_TRAIN_JOB_ID}" && -n "${ACTIVE_TRAIN_JOB_ID}" ]]; then
    LAST_TRAIN_JOB_ID="${CANDIDATE_TRAIN_JOB_ID}"
  else
    rm -f "${TRAIN_CHAIN_STATE_PATH}"
  fi
fi

SHARED_PREBUILD_JOB_ID=""
DISEASE_JOB_TOKEN="${PRIMARY_DISEASE_PROGRAM%_control}"
if [[ "${#PREBUILD_CLUSTER[@]}" -gt 0 && "${#PREBUILD_COMMANDS[@]}" -gt 0 ]]; then
  SHARED_PREBUILD_JOB_NAME="${PROJECT_PREFIX}_${DISEASE_JOB_TOKEN}_${GENERATION_STAGE}_data_prebuild"
  EXISTING_SHARED_PREBUILD=""
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    EXISTING_SHARED_PREBUILD="$(squeue -h -u "${USER}" -n "${SHARED_PREBUILD_JOB_NAME}" -o "%i" | head -n 1 || true)"
  fi
  if [[ -n "${EXISTING_SHARED_PREBUILD}" ]]; then
    SHARED_PREBUILD_JOB_ID="${EXISTING_SHARED_PREBUILD}"
  else
    SHARED_PREBUILD_CONFIG="${CONFIG_OUTPUT_ROOT}/${DISEASE_JOB_TOKEN}_${GENERATION_STAGE}_data_prebuild.yaml"
    SHARED_PREBUILD_SBATCH="${SBATCH_OUTPUT_ROOT}/${SHARED_PREBUILD_JOB_NAME}.sbatch"
    "${PYTHON_BIN}" Libs/exp/orchestration/prepare_experiment_config.py \
      --profile "${PROFILE_NAME}" \
      --output-config "${SHARED_PREBUILD_CONFIG}" \
      --base-config "${CONFIG_PATH}"

    SHARED_PREBUILD_GRES_LINE=""
    if [[ -n "${PREBUILD_CLUSTER[gres]:-}" ]]; then
      SHARED_PREBUILD_GRES_LINE="#SBATCH --gres=${PREBUILD_CLUSTER[gres]}"
    fi
    {
      echo "#!/usr/bin/env bash"
      echo "#SBATCH --partition=${PREBUILD_CLUSTER[partition]}"
      echo "#SBATCH --nodelist=${PREBUILD_CLUSTER[nodelist]}"
      echo "#SBATCH --cpus-per-task=${PREBUILD_CLUSTER[cpus_per_task]}"
      echo "#SBATCH --mem=${PREBUILD_CLUSTER[mem]}"
      if [[ -n "${SHARED_PREBUILD_GRES_LINE}" ]]; then
        echo "${SHARED_PREBUILD_GRES_LINE}"
      fi
      echo "#SBATCH --job-name=${SHARED_PREBUILD_JOB_NAME}"
      echo "#SBATCH --output=Output/logs/${SHARED_PREBUILD_JOB_NAME}.%j.out"
      echo "#SBATCH --error=Output/logs/${SHARED_PREBUILD_JOB_NAME}.%j.err"
      echo
      echo "set -euo pipefail"
      if [[ -n "${FREEZE_SIGNATURE_SHA256}" ]]; then
        echo "export PAPER_GRADE_FREEZE_SIGNATURE_SHA256=${FREEZE_SIGNATURE_SHA256}"
      fi
      echo "cd ${REPO_ROOT}"
      echo "mkdir -p Output/locks"
      echo "exec 9>Output/locks/prebuild_registry.lock"
      echo "flock 9"
      for COMMAND_TEMPLATE in "${PREBUILD_COMMANDS[@]}"; do
        COMMAND="${COMMAND_TEMPLATE//\{config_path\}/${SHARED_PREBUILD_CONFIG}}"
        COMMAND="${COMMAND//\{training_scope\}/${TRAINING_SCOPE}}"
        COMMAND="${COMMAND//\{device\}/${DEVICE_NAME}}"
        COMMAND="${COMMAND//\{disease_program\}/${DISEASE_PROGRAM}}"
        COMMAND="${COMMAND//\{python_executable\}/${PYTHON_BIN}}"
        echo "${COMMAND}"
      done
    } > "${SHARED_PREBUILD_SBATCH}"
    chmod +x "${SHARED_PREBUILD_SBATCH}"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
      echo "Prepared ${SHARED_PREBUILD_SBATCH}"
    else
      PREBUILD_SBATCH_ARGS=()
      if [[ -n "${UPSTREAM_DEPENDENCY_JOB_ID}" ]]; then
        PREBUILD_SBATCH_ARGS+=("--dependency=afterok:${UPSTREAM_DEPENDENCY_JOB_ID}")
      fi
      SHARED_PREBUILD_JOB_ID="$(sbatch --parsable "${PREBUILD_SBATCH_ARGS[@]}" "${SHARED_PREBUILD_SBATCH}")"
    fi
  fi
fi

render_command_template() {
  local template="$1"
  local config_path="$2"
  local training_scope_value="$3"
  local fold_name_value="$4"
  local window_ordinal_value="$5"
  local window_count_value="$6"
  local command="${template//\{config_path\}/${config_path}}"
  command="${command//\{training_scope\}/${training_scope_value}}"
  command="${command//\{fold_name\}/${fold_name_value}}"
  command="${command//\{device\}/${DEVICE_NAME}}"
  command="${command//\{disease_program\}/${DISEASE_PROGRAM}}"
  command="${command//\{python_executable\}/${PYTHON_BIN}}"
  command="${command//\{max_epochs\}/${SCREENING_MAX_EPOCHS}}"
  command="${command//\{patience_epochs\}/${SCREENING_PATIENCE_EPOCHS}}"
  command="${command//\{training_window_ordinal\}/${window_ordinal_value}}"
  command="${command//\{training_window_count\}/${window_count_value}}"
  printf '%s\n' "${command}"
}

safe_job_token() {
  local raw_value="$1"
  printf '%s' "${raw_value}" | sed -E 's/[^A-Za-z0-9_]+/_/g; s/_+/_/g; s/^_//; s/_$//'
}

dependency_arg_from_ids() {
  local dependency_ids=("$@")
  if [[ "${#dependency_ids[@]}" -eq 0 ]]; then
    return 0
  fi
  local dependency_value
  dependency_value="$(IFS=:; echo "${dependency_ids[*]}")"
  printf '%s\n' "--dependency=afterok:${dependency_value}"
}

write_profile_sbatch() {
  local job_name="$1"
  local sbatch_path="$2"
  local cluster_name="$3"
  shift 3
  local command_lines=("$@")
  local partition=""
  local nodelist=""
  local cpus_per_task=""
  local mem=""
  local gres_value=""
  if [[ "${cluster_name}" == "training" ]]; then
    partition="${TRAINING_CLUSTER[partition]}"
    nodelist="${TRAINING_CLUSTER[nodelist]}"
    cpus_per_task="${TRAINING_CLUSTER[cpus_per_task]}"
    mem="${TRAINING_CLUSTER[mem]}"
    gres_value="${TRAINING_CLUSTER[gres]:-}"
  elif [[ "${cluster_name}" == "evaluation" ]]; then
    partition="${EVALUATION_CLUSTER[partition]}"
    nodelist="${EVALUATION_CLUSTER[nodelist]}"
    cpus_per_task="${EVALUATION_CLUSTER[cpus_per_task]}"
    mem="${EVALUATION_CLUSTER[mem]}"
    gres_value="${EVALUATION_CLUSTER[gres]:-}"
  else
    echo "Unknown cluster profile: ${cluster_name}"
    exit 1
  fi
  {
    echo "#!/usr/bin/env bash"
    echo "#SBATCH --partition=${partition}"
    echo "#SBATCH --nodelist=${nodelist}"
    echo "#SBATCH --cpus-per-task=${cpus_per_task}"
    echo "#SBATCH --mem=${mem}"
    if [[ -n "${gres_value}" ]]; then
      echo "#SBATCH --gres=${gres_value}"
    fi
    echo "#SBATCH --job-name=${job_name}"
    echo "#SBATCH --output=Output/logs/${job_name}.%j.out"
    echo "#SBATCH --error=Output/logs/${job_name}.%j.err"
    echo
    if [[ "${cluster_name}" == "training" ]]; then
      echo "# GPU_SERIALIZATION_REQUIRED: project-internal AD screening jobs are dependency-chained because full-regimen rollout uses near-full gpu03 VRAM."
    fi
    echo "set -euo pipefail"
    if [[ -n "${FREEZE_SIGNATURE_SHA256}" ]]; then
      echo "export PAPER_GRADE_FREEZE_SIGNATURE_SHA256=${FREEZE_SIGNATURE_SHA256}"
    fi
    if [[ "${cluster_name}" == "training" && -n "${TRAINING_CLUSTER[pytorch_alloc_conf]:-}" ]]; then
      echo "export PYTORCH_ALLOC_CONF=${TRAINING_CLUSTER[pytorch_alloc_conf]}"
    fi
    if [[ "${cluster_name}" == "training" && -n "${TRAINING_CLUSTER[cublas_workspace_config]:-}" ]]; then
      echo "export CUBLAS_WORKSPACE_CONFIG=${TRAINING_CLUSTER[cublas_workspace_config]}"
    fi
    echo "cd ${REPO_ROOT}"
    for command_line in "${command_lines[@]}"; do
      echo "${command_line}"
    done
  } > "${sbatch_path}"
  chmod +x "${sbatch_path}"
}

submit_prepared_sbatch() {
  local sbatch_path="$1"
  shift
  local sbatch_args=("$@")
  local job_id
  job_id="$(sbatch --parsable "${sbatch_args[@]}" "${sbatch_path}")"
  if [[ "${SUBMISSION_THROTTLE_SECONDS}" != "0" && "${SUBMISSION_THROTTLE_SECONDS}" != "0.0" ]]; then
    sleep "${SUBMISSION_THROTTLE_SECONDS}"
  fi
  printf '%s\n' "${job_id}"
}

if [[ "${CHECKPOINTABLE_SPLIT_ENABLED}" == "True" ]]; then
  PROFILE_HAS_PRETRAINING=0
  PROFILE_HAS_EVALUATION=0
  PROFILE_HAS_ACTIVE_REPLAY=0
  PROFILE_HAS_EXPORT=0
  for COMMAND_TEMPLATE in "${TRAIN_COMMANDS[@]}"; do
    if [[ "${COMMAND_TEMPLATE}" == *"pretrain_microenvironment_world_model.py"* ]]; then
      PROFILE_HAS_PRETRAINING=1
    elif [[ "${COMMAND_TEMPLATE}" == *"evaluate_screening_system.py"* ]]; then
      PROFILE_HAS_EVALUATION=1
    elif [[ "${COMMAND_TEMPLATE}" == *"evaluate_active_learning_replay.py"* ]]; then
      PROFILE_HAS_ACTIVE_REPLAY=1
    elif [[ "${COMMAND_TEMPLATE}" == *"export_screening_bundle.py"* ]]; then
      PROFILE_HAS_EXPORT=1
    fi
  done

  SCOPE_NAMES=()
  SCOPE_FOLDS=()
  if [[ "${TRAINING_SCOPE}" == "folds" || "${TRAINING_SCOPE}" == "all" ]]; then
    for FOLD_NAME in "${FOLD_NAMES[@]}"; do
      SCOPE_NAMES+=("folds")
      SCOPE_FOLDS+=("${FOLD_NAME}")
    done
  fi
  if [[ "${TRAINING_SCOPE}" == "production" || "${TRAINING_SCOPE}" == "all" ]]; then
    SCOPE_NAMES+=("production")
    SCOPE_FOLDS+=("production")
  fi
  if [[ "${#SCOPE_NAMES[@]}" -eq 0 ]]; then
    echo "No checkpointable training scopes resolved for ${PROFILE_NAME}."
    exit 1
  fi

  PROFILE_COMPLETION_DEPENDENCIES=()
  PROFILE_MARKER_SBATCH="${SBATCH_OUTPUT_ROOT}/${PROJECT_PREFIX}_${DISEASE_JOB_TOKEN}_profile_complete_${PROFILE_NAME}.sbatch"
  for SEED_VALUE in "${SEED_VALUES[@]}"; do
    JOB_SUFFIX="${PROFILE_NAME}_seed_${SEED_VALUE}"
    OUTPUT_CONFIG="${CONFIG_OUTPUT_ROOT}/${JOB_SUFFIX}.yaml"
    "${PYTHON_BIN}" Libs/exp/orchestration/prepare_experiment_config.py \
      --profile "${PROFILE_NAME}" \
      --seed "${SEED_VALUE}" \
      --output-config "${OUTPUT_CONFIG}" \
      --base-config "${CONFIG_PATH}"

    SEED_TAIL_JOB_ID=""
    if [[ "${PROFILE_HAS_PRETRAINING}" -eq 1 ]]; then
      PRETRAIN_JOB_NAME="${PROJECT_PREFIX}_${DISEASE_JOB_TOKEN}_pretraining_${PROFILE_NAME}_seed_${SEED_VALUE}"
      PRETRAIN_SBATCH="${SBATCH_OUTPUT_ROOT}/${PRETRAIN_JOB_NAME}.sbatch"
      PRETRAIN_COMMAND="$(render_command_template "${SPLIT_COMMANDS[pretraining]}" "${OUTPUT_CONFIG}" "${TRAINING_SCOPE}" "" "" "")"
      write_profile_sbatch "${PRETRAIN_JOB_NAME}" "${PRETRAIN_SBATCH}" "training" "${PRETRAIN_COMMAND}"
      if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "Prepared ${PRETRAIN_SBATCH}"
      else
        DEPENDENCY_IDS=()
        if [[ -n "${UPSTREAM_DEPENDENCY_JOB_ID}" ]]; then
          DEPENDENCY_IDS+=("${UPSTREAM_DEPENDENCY_JOB_ID}")
        fi
        if [[ -n "${SHARED_PREBUILD_JOB_ID}" ]]; then
          DEPENDENCY_IDS+=("${SHARED_PREBUILD_JOB_ID}")
        fi
        if [[ -n "${LAST_TRAIN_JOB_ID}" ]]; then
          DEPENDENCY_IDS+=("${LAST_TRAIN_JOB_ID}")
        fi
        SBATCH_ARGS=()
        if [[ "${#DEPENDENCY_IDS[@]}" -gt 0 ]]; then
          read -r DEPENDENCY_ARG_VALUE < <(dependency_arg_from_ids "${DEPENDENCY_IDS[@]}")
          SBATCH_ARGS+=("${DEPENDENCY_ARG_VALUE}")
        fi
        PRETRAIN_JOB_ID="$(submit_prepared_sbatch "${PRETRAIN_SBATCH}" "${SBATCH_ARGS[@]}")"
        LAST_TRAIN_JOB_ID="${PRETRAIN_JOB_ID}"
        SEED_TAIL_JOB_ID="${PRETRAIN_JOB_ID}"
        echo "${LAST_TRAIN_JOB_ID}" > "${TRAIN_CHAIN_STATE_PATH}"
        echo "${LAST_TRAIN_JOB_ID}" > "${PROFILE_CHAIN_STATE_PATH}"
      fi
    fi

    for SCOPE_INDEX in "${!SCOPE_NAMES[@]}"; do
      CURRENT_SCOPE="${SCOPE_NAMES[${SCOPE_INDEX}]}"
      CURRENT_FOLD="${SCOPE_FOLDS[${SCOPE_INDEX}]}"
      CURRENT_SCOPE_TOKEN="$(safe_job_token "${CURRENT_FOLD}")"
      SCOPE_CACHE_JOB_ID=""
      if [[ "${CPU_PRECOMPUTE_ENABLED}" == "True" ]]; then
        PRECOMPUTE_JOB_NAME="${PROJECT_PREFIX}_${DISEASE_JOB_TOKEN}_cache_${CURRENT_SCOPE_TOKEN}_${PROFILE_NAME}_seed_${SEED_VALUE}"
        PRECOMPUTE_SBATCH="${SBATCH_OUTPUT_ROOT}/${PRECOMPUTE_JOB_NAME}.sbatch"
        PRECOMPUTE_COMMAND="$(render_command_template "${SPLIT_COMMANDS[cache_precompute]}" "${OUTPUT_CONFIG}" "${CURRENT_SCOPE}" "${CURRENT_FOLD}" "" "")"
        write_profile_sbatch "${PRECOMPUTE_JOB_NAME}" "${PRECOMPUTE_SBATCH}" "evaluation" "${PRECOMPUTE_COMMAND}"
        if [[ "${DRY_RUN}" -eq 1 ]]; then
          echo "Prepared ${PRECOMPUTE_SBATCH}"
        else
          PRECOMPUTE_DEPENDENCY_IDS=()
          if [[ -n "${UPSTREAM_DEPENDENCY_JOB_ID}" ]]; then
            PRECOMPUTE_DEPENDENCY_IDS+=("${UPSTREAM_DEPENDENCY_JOB_ID}")
          fi
          if [[ -n "${SHARED_PREBUILD_JOB_ID}" ]]; then
            PRECOMPUTE_DEPENDENCY_IDS+=("${SHARED_PREBUILD_JOB_ID}")
          fi
          PRECOMPUTE_SBATCH_ARGS=()
          if [[ "${#PRECOMPUTE_DEPENDENCY_IDS[@]}" -gt 0 ]]; then
            read -r PRECOMPUTE_DEPENDENCY_ARG < <(dependency_arg_from_ids "${PRECOMPUTE_DEPENDENCY_IDS[@]}")
            PRECOMPUTE_SBATCH_ARGS+=("${PRECOMPUTE_DEPENDENCY_ARG}")
          fi
          SCOPE_CACHE_JOB_ID="$(submit_prepared_sbatch "${PRECOMPUTE_SBATCH}" "${PRECOMPUTE_SBATCH_ARGS[@]}")"
        fi
      fi
      for WINDOW_INDEX in "${!WINDOW_LABELS[@]}"; do
        WINDOW_ORDINAL=$((WINDOW_INDEX + 1))
        WINDOW_LABEL="${WINDOW_LABELS[${WINDOW_INDEX}]}"
        WINDOW_TOKEN="$(safe_job_token "${WINDOW_LABEL}")"
        TRAINING_JOB_NAME="${PROJECT_PREFIX}_${DISEASE_JOB_TOKEN}_screening_${CURRENT_SCOPE_TOKEN}_${WINDOW_TOKEN}_${PROFILE_NAME}_seed_${SEED_VALUE}"
        TRAINING_SBATCH="${SBATCH_OUTPUT_ROOT}/${TRAINING_JOB_NAME}.sbatch"
        TRAINING_COMMAND="$(render_command_template "${SPLIT_COMMANDS[screening_window]}" "${OUTPUT_CONFIG}" "${CURRENT_SCOPE}" "${CURRENT_FOLD}" "${WINDOW_ORDINAL}" "${SCREENING_EPOCH_WINDOW_COUNT}")"
        write_profile_sbatch "${TRAINING_JOB_NAME}" "${TRAINING_SBATCH}" "training" "${TRAINING_COMMAND}"
        if [[ "${DRY_RUN}" -eq 1 ]]; then
          echo "Prepared ${TRAINING_SBATCH}"
          continue
        fi
        DEPENDENCY_IDS=()
        if [[ -n "${UPSTREAM_DEPENDENCY_JOB_ID}" ]]; then
          DEPENDENCY_IDS+=("${UPSTREAM_DEPENDENCY_JOB_ID}")
        fi
        if [[ -n "${SHARED_PREBUILD_JOB_ID}" ]]; then
          DEPENDENCY_IDS+=("${SHARED_PREBUILD_JOB_ID}")
        fi
        if [[ -n "${SCOPE_CACHE_JOB_ID}" ]]; then
          DEPENDENCY_IDS+=("${SCOPE_CACHE_JOB_ID}")
        fi
        if [[ -n "${LAST_TRAIN_JOB_ID}" ]]; then
          DEPENDENCY_IDS+=("${LAST_TRAIN_JOB_ID}")
        fi
        SBATCH_ARGS=()
        if [[ "${#DEPENDENCY_IDS[@]}" -gt 0 ]]; then
          read -r DEPENDENCY_ARG_VALUE < <(dependency_arg_from_ids "${DEPENDENCY_IDS[@]}")
          SBATCH_ARGS+=("${DEPENDENCY_ARG_VALUE}")
        fi
        TRAIN_JOB_ID="$(submit_prepared_sbatch "${TRAINING_SBATCH}" "${SBATCH_ARGS[@]}")"
        LAST_TRAIN_JOB_ID="${TRAIN_JOB_ID}"
        SEED_TAIL_JOB_ID="${TRAIN_JOB_ID}"
        echo "${LAST_TRAIN_JOB_ID}" > "${TRAIN_CHAIN_STATE_PATH}"
        echo "${LAST_TRAIN_JOB_ID}" > "${PROFILE_CHAIN_STATE_PATH}"
      done
    done

    if [[ "${PROFILE_HAS_EVALUATION}" -eq 1 ]]; then
      EVALUATION_JOB_NAME="${PROJECT_PREFIX}_${DISEASE_JOB_TOKEN}_evaluation_${PROFILE_NAME}_seed_${SEED_VALUE}"
      EVALUATION_SBATCH="${SBATCH_OUTPUT_ROOT}/${EVALUATION_JOB_NAME}.sbatch"
      EVALUATION_COMMAND="$(render_command_template "${SPLIT_COMMANDS[evaluation]}" "${OUTPUT_CONFIG}" "${TRAINING_SCOPE}" "" "" "")"
      write_profile_sbatch "${EVALUATION_JOB_NAME}" "${EVALUATION_SBATCH}" "evaluation" "${EVALUATION_COMMAND}"
      if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "Prepared ${EVALUATION_SBATCH}"
      else
        DEPENDENCY_IDS=()
        if [[ -n "${SEED_TAIL_JOB_ID}" ]]; then
          DEPENDENCY_IDS+=("${SEED_TAIL_JOB_ID}")
        fi
        SBATCH_ARGS=()
        if [[ "${#DEPENDENCY_IDS[@]}" -gt 0 ]]; then
          read -r DEPENDENCY_ARG_VALUE < <(dependency_arg_from_ids "${DEPENDENCY_IDS[@]}")
          SBATCH_ARGS+=("${DEPENDENCY_ARG_VALUE}")
        fi
        EVALUATION_JOB_ID="$(submit_prepared_sbatch "${EVALUATION_SBATCH}" "${SBATCH_ARGS[@]}")"
        SEED_TAIL_JOB_ID="${EVALUATION_JOB_ID}"
      fi
    fi

    if [[ "${PROFILE_HAS_ACTIVE_REPLAY}" -eq 1 ]]; then
      REPLAY_JOB_NAME="${PROJECT_PREFIX}_${DISEASE_JOB_TOKEN}_active_replay_${PROFILE_NAME}_seed_${SEED_VALUE}"
      REPLAY_SBATCH="${SBATCH_OUTPUT_ROOT}/${REPLAY_JOB_NAME}.sbatch"
      REPLAY_COMMAND="$(render_command_template "${SPLIT_COMMANDS[active_learning_replay]}" "${OUTPUT_CONFIG}" "${TRAINING_SCOPE}" "" "" "")"
      write_profile_sbatch "${REPLAY_JOB_NAME}" "${REPLAY_SBATCH}" "evaluation" "${REPLAY_COMMAND}"
      if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "Prepared ${REPLAY_SBATCH}"
      else
        DEPENDENCY_IDS=()
        if [[ -n "${SEED_TAIL_JOB_ID}" ]]; then
          DEPENDENCY_IDS+=("${SEED_TAIL_JOB_ID}")
        fi
        SBATCH_ARGS=()
        if [[ "${#DEPENDENCY_IDS[@]}" -gt 0 ]]; then
          read -r DEPENDENCY_ARG_VALUE < <(dependency_arg_from_ids "${DEPENDENCY_IDS[@]}")
          SBATCH_ARGS+=("${DEPENDENCY_ARG_VALUE}")
        fi
        REPLAY_JOB_ID="$(submit_prepared_sbatch "${REPLAY_SBATCH}" "${SBATCH_ARGS[@]}")"
        SEED_TAIL_JOB_ID="${REPLAY_JOB_ID}"
      fi
    fi

    if [[ "${PROFILE_HAS_EXPORT}" -eq 1 ]]; then
      EXPORT_JOB_NAME="${PROJECT_PREFIX}_${DISEASE_JOB_TOKEN}_export_${PROFILE_NAME}_seed_${SEED_VALUE}"
      EXPORT_SBATCH="${SBATCH_OUTPUT_ROOT}/${EXPORT_JOB_NAME}.sbatch"
      EXPORT_COMMAND="$(render_command_template "${SPLIT_COMMANDS[export]}" "${OUTPUT_CONFIG}" "${TRAINING_SCOPE}" "" "" "")"
      write_profile_sbatch "${EXPORT_JOB_NAME}" "${EXPORT_SBATCH}" "evaluation" "${EXPORT_COMMAND}"
      if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "Prepared ${EXPORT_SBATCH}"
      else
        DEPENDENCY_IDS=()
        if [[ -n "${SEED_TAIL_JOB_ID}" ]]; then
          DEPENDENCY_IDS+=("${SEED_TAIL_JOB_ID}")
        fi
        SBATCH_ARGS=()
        if [[ "${#DEPENDENCY_IDS[@]}" -gt 0 ]]; then
          read -r DEPENDENCY_ARG_VALUE < <(dependency_arg_from_ids "${DEPENDENCY_IDS[@]}")
          SBATCH_ARGS+=("${DEPENDENCY_ARG_VALUE}")
        fi
        EXPORT_JOB_ID="$(submit_prepared_sbatch "${EXPORT_SBATCH}" "${SBATCH_ARGS[@]}")"
        SEED_TAIL_JOB_ID="${EXPORT_JOB_ID}"
      fi
    fi

    if [[ "${DRY_RUN}" -eq 0 && -n "${SEED_TAIL_JOB_ID}" ]]; then
      PROFILE_COMPLETION_DEPENDENCIES+=("${SEED_TAIL_JOB_ID}")
    fi
  done

  write_profile_sbatch \
    "${PROJECT_PREFIX}_${DISEASE_JOB_TOKEN}_profile_complete_${PROFILE_NAME}" \
    "${PROFILE_MARKER_SBATCH}" \
    "evaluation" \
    "printf 'profile_complete=%s seed_count=%s\\n' '${PROFILE_NAME}' '${#SEED_VALUES[@]}'"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Prepared ${PROFILE_MARKER_SBATCH}"
    exit 0
  fi

  PROFILE_MARKER_ARGS=()
  if [[ "${#PROFILE_COMPLETION_DEPENDENCIES[@]}" -gt 0 ]]; then
    read -r PROFILE_DEPENDENCY_ARG < <(dependency_arg_from_ids "${PROFILE_COMPLETION_DEPENDENCIES[@]}")
    PROFILE_MARKER_ARGS+=("${PROFILE_DEPENDENCY_ARG}")
  fi
  PROFILE_LAST_JOB_ID="$(submit_prepared_sbatch "${PROFILE_MARKER_SBATCH}" "${PROFILE_MARKER_ARGS[@]}")"
  echo "PROFILE_LAST_JOB_ID=${PROFILE_LAST_JOB_ID}"
  echo "PROFILE_COMPLETION_DEPENDENCY_COUNT=${#PROFILE_COMPLETION_DEPENDENCIES[@]}"
  exit 0
fi

for SEED_VALUE in "${SEED_VALUES[@]}"; do
  JOB_SUFFIX="${PROFILE_NAME}_seed_${SEED_VALUE}"
  OUTPUT_CONFIG="${CONFIG_OUTPUT_ROOT}/${JOB_SUFFIX}.yaml"
  "${PYTHON_BIN}" Libs/exp/orchestration/prepare_experiment_config.py \
    --profile "${PROFILE_NAME}" \
    --seed "${SEED_VALUE}" \
    --output-config "${OUTPUT_CONFIG}" \
    --base-config "${CONFIG_PATH}"

  TRAINING_JOB_NAME="${PROJECT_PREFIX}_${DISEASE_JOB_TOKEN}_world_model_train_${JOB_SUFFIX}"
  TRAINING_SBATCH="${SBATCH_OUTPUT_ROOT}/${TRAINING_JOB_NAME}.sbatch"

  TRAINING_GRES_LINE=""
  if [[ -n "${TRAINING_CLUSTER[gres]:-}" ]]; then
    TRAINING_GRES_LINE="#SBATCH --gres=${TRAINING_CLUSTER[gres]}"
  fi
  {
    echo "#!/usr/bin/env bash"
    echo "#SBATCH --partition=${TRAINING_CLUSTER[partition]}"
    echo "#SBATCH --nodelist=${TRAINING_CLUSTER[nodelist]}"
    echo "#SBATCH --cpus-per-task=${TRAINING_CLUSTER[cpus_per_task]}"
    echo "#SBATCH --mem=${TRAINING_CLUSTER[mem]}"
    if [[ -n "${TRAINING_GRES_LINE}" ]]; then
      echo "${TRAINING_GRES_LINE}"
    fi
    echo "#SBATCH --job-name=${TRAINING_JOB_NAME}"
    echo "#SBATCH --output=Output/logs/${TRAINING_JOB_NAME}.%j.out"
    echo "#SBATCH --error=Output/logs/${TRAINING_JOB_NAME}.%j.err"
    echo
    echo "# GPU_SERIALIZATION_REQUIRED: project-internal AD screening jobs are dependency-chained because full-regimen rollout uses near-full gpu03 VRAM."
    echo "set -euo pipefail"
    if [[ -n "${FREEZE_SIGNATURE_SHA256}" ]]; then
      echo "export PAPER_GRADE_FREEZE_SIGNATURE_SHA256=${FREEZE_SIGNATURE_SHA256}"
    fi
    if [[ -n "${TRAINING_CLUSTER[pytorch_alloc_conf]:-}" ]]; then
      echo "export PYTORCH_ALLOC_CONF=${TRAINING_CLUSTER[pytorch_alloc_conf]}"
    fi
    if [[ -n "${TRAINING_CLUSTER[cublas_workspace_config]:-}" ]]; then
      echo "export CUBLAS_WORKSPACE_CONFIG=${TRAINING_CLUSTER[cublas_workspace_config]}"
    fi
    echo "cd ${REPO_ROOT}"
    for COMMAND_TEMPLATE in "${TRAIN_COMMANDS[@]}"; do
      COMMAND="${COMMAND_TEMPLATE//\{config_path\}/${OUTPUT_CONFIG}}"
      COMMAND="${COMMAND//\{training_scope\}/${TRAINING_SCOPE}}"
      COMMAND="${COMMAND//\{device\}/${DEVICE_NAME}}"
      COMMAND="${COMMAND//\{disease_program\}/${DISEASE_PROGRAM}}"
      COMMAND="${COMMAND//\{python_executable\}/${PYTHON_BIN}}"
      echo "${COMMAND}"
    done
  } > "${TRAINING_SBATCH}"
  chmod +x "${TRAINING_SBATCH}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "Prepared ${TRAINING_SBATCH}"
    continue
  fi

  SBATCH_ARGS=()
  DEPENDENCY_IDS=()
  if [[ -n "${UPSTREAM_DEPENDENCY_JOB_ID}" ]]; then
    DEPENDENCY_IDS+=("${UPSTREAM_DEPENDENCY_JOB_ID}")
  fi
  if [[ -n "${SHARED_PREBUILD_JOB_ID}" ]]; then
    DEPENDENCY_IDS+=("${SHARED_PREBUILD_JOB_ID}")
  fi
  if [[ -n "${LAST_TRAIN_JOB_ID}" ]]; then
    DEPENDENCY_IDS+=("${LAST_TRAIN_JOB_ID}")
  fi
  if [[ "${#DEPENDENCY_IDS[@]}" -gt 0 ]]; then
    DEPENDENCY_VALUE="$(IFS=:; echo "${DEPENDENCY_IDS[*]}")"
    SBATCH_ARGS+=("--dependency=afterok:${DEPENDENCY_VALUE}")
  fi
  TRAIN_JOB_ID="$(sbatch --parsable "${SBATCH_ARGS[@]}" "${TRAINING_SBATCH}")"
  LAST_TRAIN_JOB_ID="${TRAIN_JOB_ID}"
  echo "${LAST_TRAIN_JOB_ID}" > "${TRAIN_CHAIN_STATE_PATH}"
  echo "${LAST_TRAIN_JOB_ID}" > "${PROFILE_CHAIN_STATE_PATH}"
done

if [[ "${DRY_RUN}" -eq 0 && -n "${LAST_TRAIN_JOB_ID}" ]]; then
  echo "PROFILE_LAST_JOB_ID=${LAST_TRAIN_JOB_ID}"
fi
