#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
# Smoke: one protein, 4 frames, 1 rollout. Submitted by scripts/hpc_run.py, which sets
# --chdir to the immutable snapshot and --output/--error to the run's logs/.
# Required env: MDGEN_VENV, MDGEN_RESOURCE_ROOT, MDGEN_RUN_DIR (writable results/cache).
set -eo pipefail
source job/env.sh
: "${MDGEN_RESOURCE_ROOT:?}" "${MDGEN_RUN_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="$MDGEN_RUN_DIR/cache" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$XDG_CACHE_HOME"
echo "snapshot: $PWD"; nvidia-smi -L
python sim_inference.py --sim_ckpt "$MDGEN_RESOURCE_ROOT/ckpt/atlas.ckpt" \
    --data_dir "$MDGEN_RESOURCE_ROOT/data/atlas_data" \
    --split "$MDGEN_RESOURCE_ROOT/splits/mini_atlas_all.csv" \
    --pdb_id 5e3e_A --suffix _R1_i40 --num_frames 4 --num_rollouts 1 \
    --inference_seed 137 --xtc --out_dir "$MDGEN_RUN_DIR/results"
