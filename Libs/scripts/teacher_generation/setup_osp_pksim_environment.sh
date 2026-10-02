#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-Libs/configs/runtime/osp_pksim_environment.yaml}"

mapfile -t CONFIG_ASSIGNMENTS < <(python - "$CONFIG_PATH" <<'PYCONFIG'
from __future__ import annotations

import json
from pathlib import Path
import shlex
import sys

import yaml

config_path = Path(sys.argv[1])
payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
if "osp_pksim_environment" not in payload:
    raise KeyError("osp_pksim_environment section is required")
config = payload["osp_pksim_environment"]
proxy_policy = config["proxy_policy"]

def require_list(key: str) -> list[str]:
    value = config[key]
    if not isinstance(value, list) or not value:
        raise ValueError(f"osp_pksim_environment.{key} must be a non-empty list")
    return [str(item) for item in value]

def require_sequence(key: str) -> list[object]:
    value = config[key]
    if not isinstance(value, list) or not value:
        raise ValueError(f"osp_pksim_environment.{key} must be a non-empty list")
    return value

def emit(name: str, value: object) -> None:
    print(f"{name}={shlex.quote(str(value))}")

emit("ENV_NAME", config["conda_environment_name"])
emit("CONDA_NO_PLUGINS_FLAG", str(bool(config["conda_no_plugins"])).lower())
emit("CONDA_SOLVER_NAME", config["conda_solver"])
emit("CONDA_QUIET_FLAG", str(bool(config["conda_quiet"])).lower())
emit("CONDA_CHANNELS_JSON", json.dumps(require_list("conda_channels")))
emit("CONDA_PACKAGES_JSON", json.dumps(require_list("conda_packages")))
emit("CRAN_R_PACKAGES_JSON", json.dumps(require_list("cran_r_packages")))
emit("GITHUB_R_PACKAGES_JSON", json.dumps(require_sequence("github_r_packages")))
emit("REQUIRED_R_PACKAGES_JSON", json.dumps(require_list("required_r_packages")))
emit("R_PACKAGE_REPOSITORY", config["r_package_repository"])
emit("RSCRIPT_EXECUTABLE", config["rscript_executable"])
emit("UNSET_LOOPBACK_PROXY", str(bool(proxy_policy["unset_loopback_proxy_on_compute"])).lower())
emit("LOOPBACK_HOST_PATTERNS_JSON", json.dumps([str(item) for item in proxy_policy["loopback_host_patterns"]]))
emit("PROXY_ENVIRONMENT_VARIABLES_JSON", json.dumps([str(item) for item in proxy_policy["environment_variables"]]))
PYCONFIG
)
for assignment in "${CONFIG_ASSIGNMENTS[@]}"; do
  eval "$assignment"
done

if [[ "$CONDA_NO_PLUGINS_FLAG" == "true" ]]; then
  export CONDA_NO_PLUGINS=true
fi
export CONDA_SOLVER="$CONDA_SOLVER_NAME"

if [[ "$UNSET_LOOPBACK_PROXY" == "true" ]]; then
  mapfile -t PROXY_ENVIRONMENT_VARIABLES < <(python - "$PROXY_ENVIRONMENT_VARIABLES_JSON" <<'PYPROXY'
import json
import sys
for item in json.loads(sys.argv[1]):
    print(item)
PYPROXY
)
  mapfile -t LOOPBACK_HOST_PATTERNS < <(python - "$LOOPBACK_HOST_PATTERNS_JSON" <<'PYLOOPBACK'
import json
import sys
for item in json.loads(sys.argv[1]):
    print(item)
PYLOOPBACK
)
  for proxy_variable in "${PROXY_ENVIRONMENT_VARIABLES[@]}"; do
    proxy_value="${!proxy_variable-}"
    if [[ -z "$proxy_value" ]]; then
      continue
    fi
    for loopback_pattern in "${LOOPBACK_HOST_PATTERNS[@]}"; do
      if [[ "$proxy_value" == *"$loopback_pattern"* ]]; then
        unset "$proxy_variable"
        break
      fi
    done
  done
fi

mapfile -t CONDA_CHANNELS < <(python - "$CONDA_CHANNELS_JSON" <<'PYCHANNELS'
import json
import sys
for item in json.loads(sys.argv[1]):
    print(item)
PYCHANNELS
)
mapfile -t CONDA_PACKAGES < <(python - "$CONDA_PACKAGES_JSON" <<'PYPACKAGES'
import json
import sys
for item in json.loads(sys.argv[1]):
    print(item)
PYPACKAGES
)
if ! conda env list | awk '{print $1}' | grep -Fx "$ENV_NAME" >/dev/null; then
  conda_args=(create -y --solver "$CONDA_SOLVER_NAME" -n "$ENV_NAME")
else
  conda_args=(install -y --solver "$CONDA_SOLVER_NAME" -n "$ENV_NAME")
fi
if [[ "$CONDA_QUIET_FLAG" == "true" ]]; then
  conda_args+=(--quiet)
fi
for conda_channel in "${CONDA_CHANNELS[@]}"; do
  conda_args+=(-c "$conda_channel")
done
conda_args+=("${CONDA_PACKAGES[@]}")
conda "${conda_args[@]}"

export NRNSIFT_OSP_CRAN_R_PACKAGES_JSON="$CRAN_R_PACKAGES_JSON"
export NRNSIFT_OSP_GITHUB_R_PACKAGES_JSON="$GITHUB_R_PACKAGES_JSON"
export NRNSIFT_OSP_REQUIRED_R_PACKAGES_JSON="$REQUIRED_R_PACKAGES_JSON"
export NRNSIFT_OSP_R_PACKAGE_REPOSITORY="$R_PACKAGE_REPOSITORY"

R_SETUP_SCRIPT="$(mktemp "${TMPDIR:-/tmp}/nrnsift_osp_setup.XXXXXX.R")"
trap 'rm -f "$R_SETUP_SCRIPT"' EXIT
cat > "$R_SETUP_SCRIPT" <<'RS'
suppressPackageStartupMessages(library(jsonlite))
cran_packages <- fromJSON(Sys.getenv("NRNSIFT_OSP_CRAN_R_PACKAGES_JSON"))
github_packages <- fromJSON(Sys.getenv("NRNSIFT_OSP_GITHUB_R_PACKAGES_JSON"), simplifyDataFrame = TRUE)
required_packages <- fromJSON(Sys.getenv("NRNSIFT_OSP_REQUIRED_R_PACKAGES_JSON"))
repository <- Sys.getenv("NRNSIFT_OSP_R_PACKAGE_REPOSITORY")
installed_packages <- rownames(installed.packages())
missing_cran_packages <- setdiff(cran_packages, installed_packages)
if (length(missing_cran_packages) > 0) {
  install.packages(missing_cran_packages, repos = repository)
}
if (!requireNamespace("remotes", quietly = TRUE)) {
  stop("R package 'remotes' is required before installing OSPSuite-R from GitHub")
}
for (github_package_index in seq_len(nrow(github_packages))) {
  package_name <- github_packages$package_name[[github_package_index]]
  repository_name <- github_packages$repository[[github_package_index]]
  ref_name <- github_packages$ref[[github_package_index]]
  dependencies_flag <- github_packages$dependencies[[github_package_index]]
  github_reference <- if (is.na(ref_name) || ref_name == "") repository_name else paste(repository_name, ref_name, sep = "@")
  installed_packages <- rownames(installed.packages())
  if (!(package_name %in% installed_packages)) {
    remotes::install_github(github_reference, upgrade = "never", dependencies = dependencies_flag)
  }
}
installed_packages <- rownames(installed.packages())
missing_required_packages <- setdiff(required_packages, installed_packages)
if (length(missing_required_packages) > 0) {
  stop(paste("Required R packages are missing after setup:", paste(missing_required_packages, collapse = ", ")))
}
for (required_package in required_packages) {
  if (!requireNamespace(required_package, quietly = TRUE)) {
    stop(paste("Required R package cannot be loaded:", required_package))
  }
}
RS
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy conda run -n "$ENV_NAME" "$RSCRIPT_EXECUTABLE" "$R_SETUP_SCRIPT"
