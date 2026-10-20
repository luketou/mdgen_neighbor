#!/bin/bash
#SBATCH --job-name=neighbor-tests
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
# L40S: full pytest + parity takes ~45-50 min (job 48330 hit 40 min).
#SBATCH --time=01:30:00
#SBATCH --output=logs/neighbor_tests_%j.out
#SBATCH --error=logs/neighbor_tests_%j.err
set -eo pipefail
source job/env.sh
# Same default as scripts/diagnostics: a sibling ../mdgen checkout.
neighbor_resources="${MDGEN_RESOURCE_ROOT:-../mdgen}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD"
python - <<'CHECK'
import hashlib
from pathlib import Path
from mdgen.rigid_utils import rot_to_quat
print('CUDA geometry code:', rot_to_quat.__code__.co_names, flush=True)
assert 'isfinite' in rot_to_quat.__code__.co_names
print('rigid source SHA256:', hashlib.sha256(Path('mdgen/rigid_utils.py').read_bytes()).hexdigest(), flush=True)
CHECK
python -m pytest tests/test_neighbor_integration.py tests/test_ode_endpoint_storage.py tests/test_pretrained_local_core_only.py -q -s
python scripts/diagnostics/test_sdpa.py --device cuda --output_dir "results/neighbor_tests_${SLURM_JOB_ID}/sdpa"
python scripts/diagnostics/validate_ode_endpoints.py --outdir "results/neighbor_tests_${SLURM_JOB_ID}/ode"

python scripts/diagnostics/validate_neighbor_checkpoint.py --outdir "results/neighbor_tests_${SLURM_JOB_ID}/integration"
for neighbor_cli_mode in global local; do
    neighbor_cli_flags=()
    if [[ "$neighbor_cli_mode" == local ]]; then neighbor_cli_flags=(--local_spatial --no-cell_summaries); fi
    python sim_inference.py --sim_ckpt "$neighbor_resources/ckpt/atlas.ckpt" --data_dir "$neighbor_resources/data/atlas_data" --split "$neighbor_resources/splits/mini_atlas_all.csv" --pdb_id 5e3e_A --suffix _R1_i40 --num_frames 4 --num_rollouts 1 --inference_seed 137 --xtc --save_atom14 --out_dir "results/neighbor_tests_${SLURM_JOB_ID}/cli_${neighbor_cli_mode}" "${neighbor_cli_flags[@]}"
done
