#!/bin/bash
#SBATCH --job-name=NrnSift_source_inputs
#SBATCH --partition=gpu
#SBATCH --nodelist=cpu03
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/tmp/NrnSift_source_inputs_%j.out
set -euo pipefail
# Supply an explicit completed admission dependency when submitting this wrapper.
package="$1"
reuse_manifest="$2"
evidence="$3"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1
export TMPDIR="/tmp/NrnSift_source_input_runtime_${SLURM_JOB_ID}"
mkdir "$TMPDIR"
export XDG_CACHE_HOME="$TMPDIR/cache" HF_HOME="$TMPDIR/cache/hf"
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE
srun --nodes=1 --ntasks=1 --cpus-per-task=2 --cpu-bind=cores /share/home/hwxu/miniconda3/bin/python   "$package/native_source_comparison_execution.py" stage --package "$package"   --destination "/tmp/NrnSift_source_inputs_${SLURM_JOB_ID}"   --reuse-manifest "$reuse_manifest" --evidence "$evidence"
