#!/bin/bash
#SBATCH --job-name=generation-profile
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/generation_profile_%j.out
#SBATCH --error=logs/generation_profile_%j.err
# #9 Ticket 8: wall clock and NFE of one full-length FP32 rollout (the checkpoint's
# ODE sampler) of the
# ATLAS checkpoint with --local_spatial, plus a torch.profiler pass over one
# forward (kernel count, GPU busy fraction, launches from local groups).
# Synthetic 7,500-residue chain by default; extra arguments pass through, e.g.
#   MDGEN_ATLAS_CKPT=... sbatch job/submit_generation_profile.sh --group_batch_size 16
#   MDGEN_ATLAS_CKPT=... sbatch job/submit_generation_profile.sh --pdb_id NAME \
#     --data_dir D --split S --suffix _R1

set -eo pipefail
source job/env.sh
: "${MDGEN_ATLAS_CKPT:?set MDGEN_ATLAS_CKPT to the ATLAS checkpoint path}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
OUT="results/generation_profile_${SLURM_JOB_ID}"
mkdir -p "$OUT"

nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"

SOURCE=(--synthetic_residues 7500)
for arg in "$@"; do [[ "$arg" == --pdb_id* || "$arg" == --synthetic_residues* ]] && SOURCE=(); done

python -m mdgen.full_length.generation_profile \
  --sim_ckpt "$MDGEN_ATLAS_CKPT" \
  --output "$OUT/profile.json" \
  "${SOURCE[@]}" \
  "$@"

python - "$OUT/profile.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
fmt = lambda x: 'n/a' if x is None else f'{x:.3g}'
for run in report['generation']['rollouts']:
    print(f"rollout: {run['wall_clock_s']:.1f} s, NFE {run['nfe']}, {fmt(run['s_per_nfe'])} s/NFE")
p, f = report['partition'], report.get('forward_profile', {})
print(f"partition: {p['groups']} groups (median core {p['core_median']:.0f}), "
      f"{p.get('group_chunks_per_forward', 'n/a')} group chunks per forward")
print(f"forward: {report['forward_s']:.2f} s, kernels {f.get('kernel_count')}, "
      f"GPU busy {fmt(f.get('gpu_busy_fraction'))}, median kernel {fmt(f.get('kernel_median_us'))} us")
for name, region in f.get('regions', {}).items():
    print(f"  {name}: {region['launch_count']} launches ({fmt(region['launch_fraction'])})")
PY
