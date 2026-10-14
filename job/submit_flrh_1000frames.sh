#!/bin/bash
#SBATCH --job-name=mdgen2026-flrh-1000
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/flrh_1000_%j.out
#SBATCH --error=logs/flrh_1000_%j.err
#SBATCH --chdir=/share/home/r15543056/MDgen/mdgen2026

set -eo pipefail
source /etc/profile.d/modules.sh
module load python/3.11.12-gpu
module load cuda/12.8
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
source /share/home/r15543056/.venvs/mdgen2026/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
cd /share/home/r15543056/MDgen/mdgen2026

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"

python sim_inference.py \
  --sim_ckpt ckpt/forward_sim.ckpt \
  --data_dir data/4AA_data \
  --split data/demo.csv \
  --suffix _i100 \
  --num_rollouts 1 \
  --num_frames 1000 \
  --inference_seed 137 \
  --xtc \
  --out_dir "outputs/demo_flrh_${SLURM_JOB_ID}"
