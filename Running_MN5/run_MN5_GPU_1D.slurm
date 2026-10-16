#!/bin/bash
#SBATCH --job-name=cs-vm-gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=00:30:00
#SBATCH --account=<MN5_ACCOUNT>
#SBATCH --qos=<MN5_QOS>

set -euo pipefail
umask 027

mkdir -p logs

# keep threaded math libraries from over-subscribing CPU cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

cd "${SLURM_SUBMIT_DIR}"

# ---- Environment activation ----
module purge
unset PYTHONHOME
unset PYTHONPATH

# Activate environment 
if command -v mamba >/dev/null 2>&1; then
  eval "$(mamba shell hook --shell bash)"
  mamba activate CyberShake-PartitionAndVelocityModel
fi

CONFIG_FILE="inputData/parameters.in"

# GPU workflow: GPU partition + 3D interpolation
bash ./run_partition_and_velocity_model_1D_GPU_batch.sh "${CONFIG_FILE}"
