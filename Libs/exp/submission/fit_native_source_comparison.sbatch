#!/bin/bash
#SBATCH --job-name=NrnSift_source_fit
#SBATCH --partition=gpu1
#SBATCH --nodelist=gpu03
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/tmp/NrnSift_source_fit_%j.out
set -euo pipefail
# Supply the stage dependency and its exact CPU03 /tmp path.
# Direct native_wam uses the approved shared stage; matched-source fits use their scoped stage.
package="$1"
source_stage="$2"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export TMPDIR="/tmp/NrnSift_source_fit_runtime_${SLURM_JOB_ID}"
mkdir "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache" HF_HOME="$TMPDIR/cache/hf"
fit_package="$TMPDIR/package"
mkdir "$fit_package"
cp "$package/execution_chain.json" "$package/native_source_comparison_execution.py" "$package/admission_preflight.py" "$fit_package/"
/share/home/hwxu/miniconda3/bin/python - "$package" "$fit_package" <<'PY'
import json, shutil, sys
from pathlib import Path
source, target = map(Path, sys.argv[1:])
chain = json.loads((source / "execution_chain.json").read_text())
for name in chain["launcher_sha256"]:
    if not (target / name).exists(): shutil.copy2(source / name, target / name)
shutil.copy2(source / chain["code_archive_name"], target / chain["code_archive_name"])
for entry in chain["profiles"].values(): shutil.copy2(source / entry["file"], target / entry["file"])
PY
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
srun --nodes=1 --ntasks=1 --cpus-per-task=8 --cpu-bind=cores /share/home/hwxu/miniconda3/bin/python   "$fit_package/native_source_comparison_execution.py" fit --package "$fit_package"   --destination "/tmp/NrnSift_source_fit_${SLURM_JOB_ID}" --source-stage "$source_stage"
