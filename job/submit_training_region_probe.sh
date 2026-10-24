#!/bin/bash
#SBATCH --job-name=training-region-probe
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/training_region_probe_%j.out
#SBATCH --error=logs/training_region_probe_%j.err
# #9 Ticket 3 (training half): peak memory and time of one fp32-reference
# training step of the 5-layer --local_spatial ATLAS configuration on one
# Training region (--region_residues 896, --halo_rings 1 by default) x 250
# frames, cut from the frame-0 partition of a synthetic 7,500-residue chain.
# Extra train.py arguments pass through, e.g.
#   MDGEN_VENV=... sbatch job/submit_training_region_probe.sh --grad_checkpointing
# An OOM ends the job non-zero; the traceback in logs/ is the result.

set -eo pipefail
# #43 每個 Job 只跑一個分片；由 probe 保存缺 GPU 與失敗分類。
if [ "${1:-}" = --paired-shard ]; then
    shift
    export MDGEN_REQUIRE_GPU=1
    mkdir -p "${MDGEN_RUN_DIR:-$PWD}/results"
    paired_setup="$(mktemp -d "${MDGEN_RUN_DIR:-$PWD}/results/paired-setup-${SLURM_JOB_ID:-manual}-XXXXXX")"
    export MDGEN_PREFLIGHT_OUTPUT="$paired_setup/preflight.json"
    export MDGEN_DEFER_MISSING_GPU=1
    # 保持 source 內的 errexit；setup 證據不依賴 venv 或 Python 成功啟動。
    trap 'paired_exit=$?; if [ "$paired_exit" -ne 0 ]; then printf "exit=%s command=%s\n" "$paired_exit" "$BASH_COMMAND" > "$paired_setup/setup-failure.txt"; fi' EXIT
    source job/env.sh > >(tee "$paired_setup/setup.log") 2> >(tee -a "$paired_setup/setup.log" >&2)
    trap - EXIT
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
    export MODEL_DIR="$paired_setup/model"
    exec python -m mdgen.train_step_probe --matrix-shard \
        --output-root "${MDGEN_RUN_DIR:-$PWD}/results/paired" \
        --preflight-report "$MDGEN_PREFLIGHT_OUTPUT" "$@"
fi
source job/env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export MODEL_DIR="results/training_region_probe_${SLURM_JOB_ID}"
mkdir -p "$MODEL_DIR"

# Issue #43 pilot with explicit driver arguments, e.g.
#   MDGEN_ISSUE43_MODE=sweep ... job/submit_training_region_probe.sh --protein 6sup_A --replica 1 --step 64
if [ -n "${MDGEN_ISSUE43_MODE:-}${MDGEN_ISSUE43_FRAMES:-}" ]; then
    pilot=(python scripts/issue43_pilot.py
        --data_root "${MDGEN_ISSUE43_DATA:?set the Ticket 1 data root}"
        --checkpoint "${MDGEN_ATLAS_CKPT:?set the checkpoint path}" --output_dir "$MODEL_DIR")
    if [ -n "${MDGEN_ISSUE43_MODE:-}" ]; then
        exec "${pilot[@]}" --mode "$MDGEN_ISSUE43_MODE" "$@"
    fi
    exec "${pilot[@]}" --mode train --frames "$MDGEN_ISSUE43_FRAMES"
fi

nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"

python -m mdgen.train_step_probe \
  --output "$MODEL_DIR/probe.json" \
  --configs fp32-reference \
  --seed 0 \
  --synthetic_residues 7500 \
  -- \
  --train_split unused --val_split unused --data_dir unused \
  --sim_condition --prepend_ipa --num_frames 250 --batch_size 1 \
  --local_spatial --training_region \
  "$@"

python - "$MODEL_DIR/probe.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
run = report['runs'][0]
print('region:', report['batch']['regions'][0])
print('batch_shapes.trans:', run['batch_shapes']['trans'])
print(f"peak allocated {run['peak_memory_allocated_bytes'] / 1024**3:.2f} GB, "
      f"reserved {run['peak_memory_reserved_bytes'] / 1024**3:.2f} GB, "
      f"step {run['step_time_s']:.1f} s, loss_finite={run['loss_finite']}, "
      f"gradient_nonfinite={run['gradient_nonfinite']}")
PY
