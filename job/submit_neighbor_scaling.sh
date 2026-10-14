#!/bin/bash
#SBATCH --job-name=neighbor-scaling
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodelist=node-201
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/neighbor_scaling_%j.out
#SBATCH --error=logs/neighbor_scaling_%j.err
#SBATCH --chdir=/share/home/r15543056/MDgen/mdgen2026
set -eo pipefail
source /etc/profile.d/modules.sh
module load python/3.11.12-gpu cuda/12.8
source /share/home/r15543056/.venvs/mdgen/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD"
python - <<'PY'
import torch
actual=str(torch.cuda.get_device_properties(0).uuid)
assert actual=='a78cba9e-11fb-acb6-6ff5-3e12bc8ff2eb', actual
print('Verified original physical L40S:',actual,flush=True)
PY
neighbor_results="results/neighbor_scaling_${SLURM_JOB_ID}"
for neighbor_operation in forward generation; do
    for neighbor_length in 5000 7500; do
        for neighbor_mode in global local; do
            python scripts/diagnostics/scale_sdpa.py --mode "$neighbor_mode" --length "$neighbor_length" --operation "$neighbor_operation" --ffn-policy chunked --ffn-chunk-size 65536 --preprocessing native --outdir "$neighbor_results" || echo 'Error JSON retained; continuing.' >&2
        done
    done
done
python scripts/diagnostics/summarize_neighbor_scaling.py "$neighbor_results"
