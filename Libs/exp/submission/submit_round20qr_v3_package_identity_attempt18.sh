#!/usr/bin/env bash
#SBATCH --job-name=NrnSift_pkg_identity_attempt18
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=Output/audits/round20qr_v3/source_qualification_attempt18_%j.out
#SBATCH --error=Output/audits/round20qr_v3/source_qualification_attempt18_%j.err

set -euo pipefail

: "${ROUND20QR_PRODUCER_COMMIT:?set the Git-bound producer commit explicitly}"

repo_root="${SLURM_SUBMIT_DIR:?submit this script from the Git-bound repository root}"
hw_python="/share/home/anonymous/miniconda3/envs/hw/bin/python"
output_root="${repo_root}/Output/audits/round20qr_v3/source_qualification_attempt_18"
artifact_dir="${repo_root}/Input/processed/round20qr_v3/package_artifacts"
stage_root="$(mktemp -d /tmp/NrnSift_round20qr_v3_attempt18.XXXXXX)"

cleanup() {
    rm -rf "${stage_root}"
}
trap cleanup EXIT

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export JOBLIB_MULTIPROCESSING=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${stage_root}/pycache"
export PIP_NO_INDEX=1
export PIP_NO_CACHE_DIR=1

if [[ -e "${output_root}" ]]; then
    echo "refusing to overwrite append-only attempt: ${output_root}" >&2
    exit 1
fi

git -C "${repo_root}" rev-parse --is-inside-work-tree >/dev/null
git -C "${repo_root}" cat-file -e "${ROUND20QR_PRODUCER_COMMIT}^{commit}"
git clone --shared --no-checkout "${repo_root}" "${stage_root}/repo"
git -C "${stage_root}/repo" checkout --detach "${ROUND20QR_PRODUCER_COMMIT}"

"${hw_python}" -m venv "${stage_root}/package-env"
"${stage_root}/package-env/bin/python" -m pip install \
    --no-deps \
    --no-index \
    --no-compile \
    --no-cache-dir \
    --find-links "${artifact_dir}" \
    scvi-tools==0.20.3 \
    scgpt==0.2.4 \
    neuraloperator==2.0.0

if find "${stage_root}/package-env/lib" \
    \( -path "*/site-packages/scvi/*" -o -path "*/site-packages/scgpt/*" -o -path "*/site-packages/neuralop/*" \) \
    \( -path "*/__pycache__/*" -o -name "*.pyc" \) \
    -print -quit | grep -q .; then
    echo "unexpected executable bytecode in clean wheel environment" >&2
    exit 1
fi

"${hw_python}" \
    "${stage_root}/repo/Libs/scripts/builders/build_round20qr_v3_source_qualification.py" \
    --python-executable "${stage_root}/package-env/bin/python" \
    --source-root "${repo_root}" \
    --package-artifact-manifest "${stage_root}/repo/Libs/configs/experiments/round20qr_v3_package_artifact_pins.json" \
    --output-root "${output_root}" \
    --probe-timeout-seconds 45

"${hw_python}" \
    "${stage_root}/repo/Libs/scripts/validators/validate_round20qr_v3_source_qualification.py" \
    --root "${output_root}" \
    --registry "${stage_root}/repo/Libs/configs/experiments/baseline_registry.yaml"
