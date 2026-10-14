#!/bin/bash
#SBATCH --job-name=mdgen-l7500-baseline
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=logs/mdgen_l7500_baseline_%j.out
#SBATCH --error=logs/mdgen_l7500_baseline_%j.err
#SBATCH --chdir=/share/home/r15543056/MDgen/mdgen2026

set -eo pipefail
source /etc/profile.d/modules.sh
module load python/3.11.12-gpu
module load cuda/12.8
source /share/home/r15543056/.venvs/mdgen2026/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD"

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"

python scripts/diagnostics/benchmark_l_scaling.py \
  --sim_ckpt /share/home/r15543056/.cache/mdgen/atlas.ckpt \
  --output "docs/validation/mdgen2026/r2.1/C46_E2-cu128-l40s-baseline.json" \
  --lengths 256 2500 5000 7500 \
  --continue_after_oom \
  --seed 137 \
  --deterministic \
  --print_sdpa_backend "docs/validation/mdgen2026/r2.1/C46_E2-cu128-l40s-sdpa-backend.txt"
