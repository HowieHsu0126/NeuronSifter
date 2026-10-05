#!/usr/bin/env bash
#SBATCH --job-name=NrnSift_hw_env_attempt24
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=Output/audits/round20qr_v3/source_qualification_attempt24_%j.out
#SBATCH --error=Output/audits/round20qr_v3/source_qualification_attempt24_%j.err

set -euo pipefail

repo_root="${SLURM_SUBMIT_DIR:?submit this script from the repository root}"
hw_python="/share/home/anonymous/miniconda3/envs/hw/bin/python"
output_root="${repo_root}/Output/audits/round20qr_v3/source_qualification_attempt_24"
stage_root="$(mktemp -d /tmp/NrnSift_round20qr_v3_attempt24.XXXXXX)"

cleanup() {
    rm -rf "${stage_root}"
}
trap cleanup EXIT

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export JOBLIB_MULTIPROCESSING=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${stage_root}/pycache"

if [[ -e "${output_root}" ]]; then
    echo "refusing to overwrite append-only attempt: ${output_root}" >&2
    exit 1
fi

"${hw_python}" - <<'PY' > "${stage_root}/environment_preflight.json"
import importlib.metadata
import importlib.util
import json
import sys

packages = {
    "scvi": ("scvi-tools", "scvi"),
    "scgpt": ("scgpt", "scgpt"),
    "uce": ("uce", "uce"),
    "fno_neuraloperator": ("neuraloperator", "neuralop"),
}
rows = []
for baseline_uid, (distribution_name, module_name) in packages.items():
    row = {
        "baseline_uid": baseline_uid,
        "distribution": distribution_name,
        "module": module_name,
        "distribution_present": False,
        "version": "",
        "distribution_path": "",
        "project_urls": [],
        "license": "",
        "module_spec_origin": "",
        "import_smoke": "not_run",
        "import_error": "",
    }
    try:
        distribution = importlib.metadata.distribution(distribution_name)
        metadata = distribution.metadata
        row.update(
            {
                "distribution_present": True,
                "version": distribution.version,
                "distribution_path": str(distribution._path),
                "project_urls": metadata.get_all("Project-URL") or [],
                "license": metadata.get("License", ""),
            }
        )
    except importlib.metadata.PackageNotFoundError:
        row["import_error"] = "distribution_not_found"
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        row["module_spec_origin"] = spec.origin or ""
        try:
            __import__(module_name)
            row["import_smoke"] = "passed"
        except BaseException as exc:  # Preserve import failures without halting the matrix.
            row["import_smoke"] = "failed"
            row["import_error"] = f"{type(exc).__name__}: {exc}"[:500]
    elif not row["import_error"]:
        row["import_error"] = "module_not_found"
    rows.append(row)

print(
    json.dumps(
        {
            "schema_version": "neuronsifter-round20qr-v3-hw-environment-preflight-v1",
            "python_executable": sys.executable,
            "rows": rows,
        },
        indent=2,
        sort_keys=True,
    )
)
PY

mkdir -p "${output_root}"
cp "${stage_root}/environment_preflight.json" "${output_root}/environment_preflight.json"
"${hw_python}" - <<'PY' "${output_root}/environment_preflight.json"
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["artifact_uid"] = hashlib.sha256(
    json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "environment_preflight=${output_root}/environment_preflight.json"
