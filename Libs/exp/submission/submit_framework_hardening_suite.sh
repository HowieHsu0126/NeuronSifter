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

readarray -t PROFILE_VALUES < <(CONFIG_PATH="${CONFIG_PATH}" "${PYTHON_BIN}" - <<'PY'
import os
from Libs.utils.data.raw_data_registry import load_config

config = load_config(os.environ["CONFIG_PATH"])
cluster_registry = config["cluster_registry"]
if "active_learning" not in cluster_registry:
    raise ValueError("cluster_registry.active_learning is required for framework hardening submission.")
suite = config["paper"]["publication_claim_suite"]
print(f"suite::framework_audit_enabled::{suite.get('framework_audit_enabled', False)}")
for profile_name in suite.get("framework_audit_profiles", []):
    print(f"profile::{profile_name}")
PY
)

FRAMEWORK_AUDIT_ENABLED=0
AUDIT_PROFILES=()
for value in "${PROFILE_VALUES[@]}"; do
  if [[ "${value}" == suite::framework_audit_enabled::* ]]; then
    if [[ "${value##*::}" == "True" || "${value##*::}" == "true" ]]; then
      FRAMEWORK_AUDIT_ENABLED=1
    fi
  elif [[ "${value}" == profile::* ]]; then
    AUDIT_PROFILES+=("${value#profile::}")
  fi
done

if [[ "${FRAMEWORK_AUDIT_ENABLED}" -eq 0 ]]; then
  echo "AUDIT_LAST_JOB_ID="
  exit 0
fi
if [[ "${#AUDIT_PROFILES[@]}" -eq 0 ]]; then
  echo "publication_claim_suite.framework_audit_profiles is empty."
  exit 1
fi

DEPENDENCY_ARGS=()
if [[ -n "${UPSTREAM_DEPENDENCY_JOB_ID}" ]]; then
  DEPENDENCY_ARGS=("--dependency-job-id" "${UPSTREAM_DEPENDENCY_JOB_ID}")
fi
PRINT_ONLY_ARG=()
if [[ "${DRY_RUN}" -eq 1 ]]; then
  PRINT_ONLY_ARG=("--print-only")
fi

LAST_AUDIT_JOB_ID=""
for profile_name in "${AUDIT_PROFILES[@]}"; do
  output="$(bash "${SCRIPT_DIR}/submit_framework_hardening_profile.sh" "${profile_name}" --config "${CONFIG_PATH}" "${PRINT_ONLY_ARG[@]}" "${DEPENDENCY_ARGS[@]}")"
  printf '%s\n' "${output}"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    LAST_AUDIT_JOB_ID="$(printf '%s\n' "${output}" | awk -F= '$1 == "AUDIT_PROFILE_JOB_ID" {print $2}' | tail -n 1)"
    if [[ -z "${LAST_AUDIT_JOB_ID}" ]]; then
      echo "Failed to parse AUDIT_PROFILE_JOB_ID for ${profile_name}."
      exit 1
    fi
    DEPENDENCY_ARGS=("--dependency-job-id" "${LAST_AUDIT_JOB_ID}")
  fi
done

if [[ "${DRY_RUN}" -eq 0 ]]; then
  echo "AUDIT_LAST_JOB_ID=${LAST_AUDIT_JOB_ID}"
fi
