#!/bin/bash
#SBATCH --job-name=halo-error
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/halo_error_%j.out
#SBATCH --error=logs/halo_error_%j.err
# #12 Ticket 5: loss-core output error of Training regions with --halo_rings 1
# and 2 against the full-protein forward of the ATLAS checkpoint, under the same
# noise, t and loss groups. Pick proteins whose full forward fits on the GPU.
#   MDGEN_ATLAS_CKPT=... MDGEN_HALO_DATA=dir MDGEN_HALO_SPLIT=split.csv \
#     sbatch job/submit_halo_error.sh --pdb_id NAME --replicate 1 [--partition_dir P]
# Further mdgen.full_length.halo_error arguments pass through.

set -eo pipefail
source job/env.sh
: "${MDGEN_ATLAS_CKPT:?set MDGEN_ATLAS_CKPT to the ATLAS checkpoint path}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
OUT="results/halo_error_${SLURM_JOB_ID}"
mkdir -p "$OUT"

# Ticket 3/4: MDGEN_ISSUE43_MODE=baseline_forward or long_forward; pass --cases or shard options.
# Use a 30-minute scheduler limit for the frozen matrix. Planning calls the driver directly.
if [ -n "${MDGEN_ISSUE43_FRAMES:-}" ] || [ "${MDGEN_ISSUE43_MODE:-}" = baseline_forward ] \
        || [ "${MDGEN_ISSUE43_MODE:-}" = long_forward ]; then
    frame_args=()
    if [ -n "${MDGEN_ISSUE43_FRAMES:-}" ]; then
        frame_args=(--frames "$MDGEN_ISSUE43_FRAMES")
    fi
    exec python scripts/issue43_pilot.py --mode "${MDGEN_ISSUE43_MODE:-forward}" \
        "${frame_args[@]}" \
        --data_root "${MDGEN_ISSUE43_DATA:?set the Ticket 1 data root}" \
        --checkpoint "$MDGEN_ATLAS_CKPT" --output_dir "$OUT" "$@"
fi

: "${MDGEN_HALO_DATA:?set MDGEN_HALO_DATA to the NPY directory}"
: "${MDGEN_HALO_SPLIT:?set MDGEN_HALO_SPLIT to the split CSV}"

nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"

python -m mdgen.full_length.halo_error \
  --sim_ckpt "$MDGEN_ATLAS_CKPT" \
  --data_dir "$MDGEN_HALO_DATA" \
  --split "$MDGEN_HALO_SPLIT" \
  --output "$OUT/halo_error.json" \
  --halo_rings 1 2 \
  "$@"

python - "$OUT/halo_error.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))['summary']
fmt = lambda x: 'n/a' if x is None else f'{x:.3g}'
for k, entry in summary.items():
    rel = entry['relative']
    print(f"halo_rings={k}: relative median {fmt(rel['median'])}, max {fmt(rel['max'])} "
          f"over {rel['count']} loss residues")
    for ring, by in entry['by_ring'].items():
        print(f"  ring {ring}: relative median {fmt(by['relative']['median'])}, "
              f"max {fmt(by['relative']['max'])} ({by['relative']['count']})")
PY
