#!/bin/bash
#SBATCH --job-name=atlas-validation-cpu
#SBATCH --partition=cpu-2g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
export MDGEN_MODULES="python/3.11.12-gpu"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" MDGEN_REQUIRE_GPU=0
source job/env.sh
echo "host=$(hostname) job=${SLURM_JOB_ID} python=$(command -v python)"
exec "$@"
