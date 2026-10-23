#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${MDGEN_PYTHON:-${MDGEN_CPU_VENV:-$repo_root/.venv-cpu}/bin/python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo 'CPU environment is missing. Run bash scripts/bootstrap_cpu.sh first, or set MDGEN_PYTHON.' >&2
    exit 1
fi
"$python_bin" -c 'import pathlib, sys; expected = tuple(map(int, pathlib.Path(".python-version").read_text().strip().split("."))); sys.exit(0 if sys.version_info[:2] == expected else f"Use Python {expected[0]}.{expected[1]} to match CI")'
"$python_bin" scripts/check_repository.py
bash scripts/check_git_whitespace.sh
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/diagnostics/gpu_preflight.py --output results/cpu-preflight.json
"$python_bin" -m pytest tests -q
mkdir -p results/sdpa
sdpa_output="$(mktemp -d results/sdpa/check-XXXXXXXX)"
"$python_bin" scripts/diagnostics/test_sdpa.py --device cpu --output_dir "$sdpa_output"
