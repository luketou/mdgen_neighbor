#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_version="$(tr -d '\r\n' < .python-version)"
python_bin="${MDGEN_PYTHON:-python$python_version}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    if [[ -z "${MDGEN_PYTHON:-}" ]] && command -v uv >/dev/null 2>&1; then
        python_bin="$(uv python find "$python_version" 2>/dev/null || true)"
    fi
    if [[ -z "$python_bin" ]] || ! command -v "$python_bin" >/dev/null 2>&1; then
        echo "Python $python_version is required. Install it (e.g. uv python install $python_version), then rerun, or set MDGEN_PYTHON to its executable." >&2
        exit 1
    fi
fi
check_version() {
    "$1" -c 'import sys; expected = tuple(map(int, sys.argv[1].split("."))); actual = sys.version_info[:2]; sys.exit(0 if actual == expected else f"Expected Python {sys.argv[1]}, got {actual[0]}.{actual[1]}")' "$python_version"
}
check_version "$python_bin"
venv_dir="${MDGEN_CPU_VENV:-$repo_root/.venv-cpu}"
if [[ ! -f "$venv_dir/pyvenv.cfg" ]]; then
    if [[ -e "$venv_dir" ]]; then
        echo "Refusing to replace existing non-venv path: $venv_dir" >&2
        exit 1
    fi
    "$python_bin" -m venv "$venv_dir"
fi
venv_python="$venv_dir/bin/python"
check_version "$venv_python"

# Install the CPU wheel first; installing requirements concurrently can select CUDA wheels.
torch_requirement="$(sed -n '/^torch==/p' requirements.txt)"
"$venv_python" -m pip install --disable-pip-version-check "$torch_requirement+cpu" --index-url https://download.pytorch.org/whl/cpu
"$venv_python" -m pip install --disable-pip-version-check -r requirements-dev.txt
"$venv_python" -m pip check
if ! git var GIT_AUTHOR_IDENT >/dev/null 2>&1; then
    echo 'Git author identity is not configured. Before committing, set repo-local user.name and user.email with git config --local.' >&2
fi
if ! command -v gh >/dev/null 2>&1; then
    echo 'gh is unavailable; GitHub connector fallback is documented in docs/agents/tools.md.'
fi
printf 'CPU environment ready: %s\nRun checks: MDGEN_CPU_VENV=%q bash scripts/check.sh\n' "$venv_python" "$venv_dir"
