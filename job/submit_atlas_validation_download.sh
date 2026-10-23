#!/bin/bash
#SBATCH --job-name=atlas-test3-download
#SBATCH --partition=cpu-2g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:00:00
set -euo pipefail
echo "host=$(hostname) job=${SLURM_JOB_ID} started=$(date -u +%FT%TZ)"
export MDGEN_MODULES="python/3.11.12-gpu" MDGEN_REQUIRE_GPU=0
source job/env.sh
exec python -u scripts/install_atlas_mirror.py \
  --manifest scripts/atlas_validation_mirror_manifest.json \
  --destination /share/home/r15543056/MDgen/mdgen/data/atlas_validation_sims \
  --run-dir "${MDGEN_RUN_DIR:?}"
