#!/bin/bash
#SBATCH --job-name=mdgen-flrh-geometry
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/flrh_geometry_%j.out
#SBATCH --error=logs/flrh_geometry_%j.err

set -eo pipefail
source job/env.sh
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"

python sim_inference.py \
  --sim_ckpt ckpt/forward_sim.ckpt \
  --data_dir data/4AA_data \
  --split data/demo.csv \
  --suffix _i100 \
  --num_rollouts 10 \
  --num_frames 1000 \
  --inference_seed 137 \
  --xtc --save_atom14 \
  --out_dir "outputs/flrh_geometry_${SLURM_JOB_ID}/generated"

RUN="outputs/flrh_geometry_${SLURM_JOB_ID}"
python scripts/analyze_flrh_basic.py --generated "$RUN/generated/FLRH" --outdir "$RUN/analysis" > "$RUN/basic.log" 2>&1
python -m scripts.diagnose_flrh_geometry --generated "$RUN/generated/FLRH" --outdir "$RUN/analysis" > "$RUN/diagnosis.log" 2>&1
