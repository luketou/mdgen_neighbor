#!/bin/bash
#SBATCH --job-name=mdgen-flrh-tolerance
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/flrh_tolerance_%j.out
#SBATCH --error=logs/flrh_tolerance_%j.err

set -eo pipefail
source job/env.sh
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"


ROOT="outputs/flrh_tolerance_${SLURM_JOB_ID}"
mkdir -p "$ROOT"
for LEVEL in baseline tighter tightest; do
  case "$LEVEL" in
    baseline) ATOL=1e-6; RTOL=1e-3 ;;
    tighter) ATOL=1e-7; RTOL=1e-4 ;;
    tightest) ATOL=1e-8; RTOL=1e-5 ;;
  esac
  RUN="$ROOT/$LEVEL"
  mkdir -p "$RUN/generated" "$RUN/analysis"
  python sim_inference.py --sim_ckpt ckpt/forward_sim.ckpt --data_dir data/4AA_data --split data/demo.csv --suffix _i100 --num_rollouts 10 --num_frames 1000 --inference_seed 137 --xtc --ode_atol "$ATOL" --ode_rtol "$RTOL" --out_dir "$RUN/generated" > "$RUN/inference.log" 2>&1
  python scripts/analyze_flrh_basic.py --generated "$RUN/generated/FLRH" --outdir "$RUN/analysis" > "$RUN/basic.log" 2>&1
  python -m scripts.analyze_peptide_sim --mddir data/4AA_sims --pdbdir "$RUN/generated" --plot --save --num_workers 1 > "$RUN/pyemma.log" 2>&1
  python -m scripts.plot_flrh_comparison --generated "$RUN/generated/FLRH" --outdir "$RUN/analysis" --pyemma "$RUN/generated/out.pkl" > "$RUN/plot.log" 2>&1
  echo "Completed $LEVEL"
done
