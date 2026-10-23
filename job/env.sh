# Shared setup sourced by job/*.sh. Submit from the repository root:
#   MDGEN_VENV=/path/to/venv sbatch job/<script>.sh
# MDGEN_MODULES overrides the modules to load (default below); set it to an
# empty string to skip `module load` on machines without environment modules.
if [ ! -f mdgen/wrapper.py ]; then
    echo "job/env.sh: run sbatch from the repository root (cwd: $PWD)" >&2
    exit 1
fi
: "${MDGEN_VENV:?set MDGEN_VENV to the virtualenv used for this repository}"
# sbatch from an SSH session exports SSH_CLIENT, so the `bash -c` inside
# `module load auto_gpu` sources ~/.bashrc, whose mkdir of the login node's
# /run/user/<uid> fails on compute nodes and aborts the module load (49304).
unset SSH_CLIENT SSH_CONNECTION SSH_TTY
export XDG_RUNTIME_DIR="/tmp/xdg-$USER"
mkdir -p "$XDG_RUNTIME_DIR"
# mdgen/logger.py creates $MODEL_DIR (default ./workdir/default) on import;
# a read-only snapshot needs it outside the source tree (49313).
if [ -z "${MODEL_DIR:-}" ] && [ ! -w . ]; then
    export MODEL_DIR="${MDGEN_RUN_DIR:-$PWD/..}/workdir/${SLURM_JOB_ID:-$$}"
fi
mdgen_modules="${MDGEN_MODULES-python/3.11.12-gpu cuda/12.8}"
if [ -n "$mdgen_modules" ]; then
    source /etc/profile.d/modules.sh
    module load $mdgen_modules
fi
source "$MDGEN_VENV/bin/activate"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Some nodes (e.g. node-201) confine a one-GPU job to its card as device 0 but
# still set CUDA_VISIBLE_DEVICES to the host index, which fails with
# CUDA_ERROR_NO_DEVICE on the second card. GPU_DEVICE_ORDINAL is the in-job index.
if [ -n "${GPU_DEVICE_ORDINAL:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_DEVICE_ORDINAL"
fi
