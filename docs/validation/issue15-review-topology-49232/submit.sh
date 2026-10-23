#!/bin/bash
#SBATCH --job-name=pr19-fix
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=results/pytest-%j.out
#SBATCH --error=results/pytest-%j.err
set -eo pipefail
run_root="$PWD"
trap 'printf "EXIT_CODE=%s\n" "$?" > "$run_root/results/completion.txt"' EXIT
export XDG_RUNTIME_DIR=/tmp/mdgen-runtime-$USER
mkdir -p "$XDG_RUNTIME_DIR"
source /etc/profile.d/modules.sh
module load python/3.11.12-gpu
source /share/home/r15543056/.venvs/mdgen/bin/activate
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$run_root/source"
export PYTHONPATH="$PWD"
sha256sum -c manifest.sha256 > "$run_root/results/manifest-check.txt"
python -c 'import sys, torch; print(sys.version); print(torch.__version__)'
python -m pytest tests/test_dataset_training_region.py -k "${MDGEN_TEST_FILTER:-region}" -q -p no:cacheprovider --junitxml="$run_root/results/dataset.xml"
if [[ "${MDGEN_FULL_SUITE:-0}" == 1 ]]; then
    python -m pytest tests -q -p no:cacheprovider --junitxml="$run_root/results/full-suite.xml"
fi
