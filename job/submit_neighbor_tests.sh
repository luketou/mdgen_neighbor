#!/bin/bash
#SBATCH --job-name=neighbor-tests
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodelist=node-201
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=00:20:00
#SBATCH --output=logs/neighbor_tests_%j.out
#SBATCH --error=logs/neighbor_tests_%j.err
#SBATCH --chdir=/share/home/r15543056/MDgen/mdgen2026
set -eo pipefail
source /etc/profile.d/modules.sh
module load python/3.11.12-gpu cuda/12.8
source /share/home/r15543056/.venvs/mdgen/bin/activate
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
python test_sdpa.py --device cuda --output_dir "results/neighbor_tests_${SLURM_JOB_ID}/sdpa"
python scripts/diagnostics/validate_ode_endpoints.py --outdir "results/neighbor_tests_${SLURM_JOB_ID}/ode"

python scripts/diagnostics/validate_neighbor_checkpoint.py --outdir "results/neighbor_tests_${SLURM_JOB_ID}/integration"
for neighbor_cli_mode in global local; do
    neighbor_cli_flags=()
    if [[ "$neighbor_cli_mode" == local ]]; then neighbor_cli_flags=(--local_spatial --no-cell_summaries); fi
    python sim_inference.py --sim_ckpt ../mdgen/ckpt/atlas.ckpt --data_dir ../mdgen/data/atlas_data --split ../mdgen/splits/mini_atlas_all.csv --pdb_id 5e3e_A --suffix _R1_i40 --num_frames 4 --num_rollouts 1 --inference_seed 137 --xtc --save_atom14 --out_dir "results/neighbor_tests_${SLURM_JOB_ID}/cli_${neighbor_cli_mode}" "${neighbor_cli_flags[@]}"
done
