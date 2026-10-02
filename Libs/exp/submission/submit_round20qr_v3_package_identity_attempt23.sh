#!/usr/bin/env bash
#SBATCH --job-name=NrnSift_pkg_identity_attempt23
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu04
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=Output/audits/round20qr_v3/source_qualification_attempt23_%j.out
#SBATCH --error=Output/audits/round20qr_v3/source_qualification_attempt23_%j.err

set -euo pipefail

: "${ROUND20QR_PRODUCER_COMMIT:?set the Git-bound producer commit explicitly}"

repo_root="${SLURM_SUBMIT_DIR:?submit this script from the Git-bound repository root}"
hw_python="/share/home/hwxu/miniconda3/envs/hw/bin/python"
output_root="${repo_root}/Output/audits/round20qr_v3/source_qualification_attempt_23"
artifact_dir="${repo_root}/Input/processed/round20qr_v3/package_artifacts"
stage_root="$(mktemp -d /tmp/NrnSift_round20qr_v3_attempt23.XXXXXX)"
source_root="${stage_root}/source"
wheel_cache="${stage_root}/wheel-cache"

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
git -C "${stage_root}/repo" sparse-checkout init --cone
git -C "${stage_root}/repo" sparse-checkout set \
    Libs/scripts/builders \
    Libs/scripts/validators \
    Libs/configs/experiments
git -C "${stage_root}/repo" checkout --detach "${ROUND20QR_PRODUCER_COMMIT}"

mkdir -p "${source_root}/Libs/external/neural_operator" "${wheel_cache}"
for checkout_name in Transolver_plus wdno; do
    source_checkout="${repo_root}/Libs/external/neural_operator/${checkout_name}"
    staged_checkout="${source_root}/Libs/external/neural_operator/${checkout_name}"
    checkout_commit="$(git -C "${source_checkout}" rev-parse HEAD)"
    checkout_remote="$(git -C "${source_checkout}" remote get-url origin)"
    git clone --shared --no-checkout "${source_checkout}" "${staged_checkout}"
    git -C "${staged_checkout}" remote set-url origin "${checkout_remote}"
    git -C "${staged_checkout}" checkout --detach "${checkout_commit}"
done

for relative_path in \
    __pycache__/main_airplane.cpython-313.pyc \
    __pycache__/train_airplane.cpython-313.pyc \
    dataset/__pycache__/dataset.cpython-313.pyc \
    models/__pycache__/Transolver_plus.cpython-313.pyc; do
    source_file="${repo_root}/Libs/external/neural_operator/Transolver_plus/${relative_path}"
    staged_file="${source_root}/Libs/external/neural_operator/Transolver_plus/${relative_path}"
    [[ -f "${source_file}" ]]
    mkdir -p "$(dirname "${staged_file}")"
    cp -p "${source_file}" "${staged_file}"
done

for wheel_name in \
    scvi_tools-0.20.3-py3-none-any.whl \
    scgpt-0.2.4-py3-none-any.whl \
    neuraloperator-2.0.0-py3-none-any.whl; do
    cp -p "${artifact_dir}/${wheel_name}" "${wheel_cache}/${wheel_name}"
done

"${hw_python}" -m venv "${stage_root}/package-env"
package_python="${stage_root}/package-env/bin/python"
package_site="$("${package_python}" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
"${package_python}" -m pip install \
    --no-deps \
    --no-index \
    --no-compile \
    --no-cache-dir \
    --find-links "${wheel_cache}" \
    scvi-tools==0.20.3 \
    scgpt==0.2.4 \
    neuraloperator==2.0.0

# Pip installation is retained as a reproducible preflight; final identity is
# materialized directly from the verified wheel so RECORD remains byte-exact.
"${package_python}" -m pip uninstall -y scvi-tools scgpt neuraloperator
for wheel_name in \
    scvi_tools-0.20.3-py3-none-any.whl \
    scgpt-0.2.4-py3-none-any.whl \
    neuraloperator-2.0.0-py3-none-any.whl; do
    "${package_python}" -m zipfile -e "${wheel_cache}/${wheel_name}" "${package_site}"
done

if find "${package_site}" \
    \( -path "*/scvi/*" -o -path "*/scgpt/*" -o -path "*/neuralop/*" \) \
    \( -path "*/__pycache__/*" -o -name "*.pyc" \) \
    -print -quit | grep -q .; then
    echo "unexpected executable bytecode in exact wheel environment" >&2
    exit 1
fi

"${hw_python}" \
    "${stage_root}/repo/Libs/scripts/builders/build_round20qr_v3_source_qualification.py" \
    --python-executable "${package_python}" \
    --source-root "${source_root}" \
    --package-artifact-manifest "${stage_root}/repo/Libs/configs/experiments/round20qr_v3_package_artifact_pins.json" \
    --package-artifact-cache-root "${wheel_cache}" \
    --output-root "${output_root}" \
    --probe-timeout-seconds 45

"${hw_python}" \
    "${stage_root}/repo/Libs/scripts/validators/validate_round20qr_v3_source_qualification.py" \
    --root "${output_root}" \
    --registry "${stage_root}/repo/Libs/configs/experiments/baseline_registry.yaml"
