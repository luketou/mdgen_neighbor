# Shared setup sourced by job/*.sh. Submit from the repository root:
#   MDGEN_VENV=/path/to/venv sbatch job/<script>.sh
# MDGEN_MODULES overrides the modules to load (default below); set it to an
# empty string to skip `module load` on machines without environment modules.
if [ ! -f mdgen/wrapper.py ]; then
    echo "job/env.sh: run sbatch from the repository root (cwd: $PWD)" >&2
    exit 1
fi
: "${MDGEN_VENV:?set MDGEN_VENV to the virtualenv used for this repository}"
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
# Use the activated project's interpreter for the shared readiness gate.
"$MDGEN_VENV/bin/python" scripts/diagnostics/gpu_preflight.py --require-gpu \
    --output "${MDGEN_PREFLIGHT_OUTPUT:-results/preflight_${SLURM_JOB_ID:-manual}.json}" || return $?
