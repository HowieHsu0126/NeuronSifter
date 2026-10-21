#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
PROJECT_PREFIX="$(tr -d '[:space:]' < PROJECT_PREFIX)"
USER_NAME="${USER:-$(id -un)}"
squeue -u "${USER_NAME}" -n "${PROJECT_PREFIX}*"
