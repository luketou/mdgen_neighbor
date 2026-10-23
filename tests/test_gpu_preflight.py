"""GPU readiness is reported through the public CLI, including its exit code."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_cpu_runner_skips_optional_gpu_check(tmp_path):
    report = tmp_path / 'report.json'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/diagnostics/gpu_preflight.py'),
                             '--output', str(report)],
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'}, capture_output=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(report.read_text())
    assert data['status'] == 'skipped'
    assert data['passed'] is False


def test_required_gpu_cannot_silently_skip_on_cpu(tmp_path):
    report = tmp_path / 'report.json'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/diagnostics/gpu_preflight.py'),
                             '--require-gpu', '--output', str(report)],
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'}, capture_output=True)
    assert result.returncode == 1
    data = json.loads(report.read_text())
    assert data['status'] == 'failed'
    assert data['passed'] is False
    assert data['error']


def test_allocated_slurm_gpu_requires_real_cuda_even_without_flag(tmp_path):
    report = tmp_path / 'allocated.json'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/diagnostics/gpu_preflight.py'),
                             '--output', str(report)],
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'SLURM_JOB_GPUS': '0'},
                            capture_output=True)
    assert result.returncode == 1
    assert json.loads(report.read_text())['status'] == 'failed'
