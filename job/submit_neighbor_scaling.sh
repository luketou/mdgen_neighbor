#!/bin/bash
#SBATCH --job-name=neighbor-scaling
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/neighbor_scaling_%j.out
#SBATCH --error=logs/neighbor_scaling_%j.err
set -eo pipefail
source job/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$PWD"
python - <<'PY'
import torch
import os
actual=str(torch.cuda.get_device_properties(0).uuid)
# The recorded comparison ran on one physical L40S; pin it to reproduce that.
expected=os.environ.get('MDGEN_EXPECTED_GPU_UUID')
if expected:
    assert actual==expected, actual
    print('Verified expected physical GPU:',actual,flush=True)
else:
    print('GPU UUID (unpinned):',actual,flush=True)
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
