#!/bin/bash
#SBATCH --job-name=pytest
#SBATCH --partition=l40s
#SBATCH --account=l40s
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pytest_%j.out
#SBATCH --error=logs/pytest_%j.err
# pytest on the compute node; arguments pass to pytest (default: the whole suite), e.g.
#   python scripts/hpc_run.py submit --tag pytest job/submit_pytest.sh tests/test_halo_error.py
# The cache provider is off because a hpc_run.py snapshot is read-only.

set -eo pipefail
source job/env.sh
export PYTHONUNBUFFERED=1
[ $# -gt 0 ] || set -- tests
python -m pytest -p no:cacheprovider -q "$@"
