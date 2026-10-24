"""Exercise paired launcher setup with fake modules/interpreter; never submit jobs."""
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('failure', ['activation', 'module', 'missing_venv', 'wrong_root', 'preflight_missing', 'preflight_program'])
def test_paired_launcher_stops_setup_failures_but_records_missing_gpu(tmp_path, failure):
    (tmp_path / 'job').mkdir()
    (tmp_path / 'mdgen').mkdir()
    (tmp_path / 'mdgen/wrapper.py').touch()
    for name in ('env.sh', 'submit_training_region_probe.sh'):
        (tmp_path / 'job' / name).write_text((ROOT / 'job' / name).read_text())
    venv = tmp_path / 'venv'
    (venv / 'bin').mkdir(parents=True)
    activate = f'export PATH="{venv}/bin:$PATH"\n'
    if failure == 'activation':
        activate += 'false\ntouch activation-continued\n'
    (venv / 'bin/activate').write_text(activate)
    fake_python = venv / 'bin/python'
    fake_python.write_text('''#!/bin/bash
if [ "$1" = scripts/diagnostics/gpu_preflight.py ]; then
    printf '{"passed":false,"failure_category":"%s","error":"test"}' "$TEST_CATEGORY" > "$MDGEN_PREFLIGHT_OUTPUT"
    exit 1
fi
if [ "$1" = -c ]; then
    exec "$TEST_REAL_PYTHON" "$@"
fi
touch shard-started
''')
    fake_python.chmod(0o755)
    import sys
    env = {**os.environ, 'MDGEN_VENV': str(venv), 'MDGEN_MODULES': '',
           'MDGEN_RUN_DIR': str(tmp_path), 'TEST_REAL_PYTHON': sys.executable,
           'TEST_CATEGORY': 'missing_gpu' if failure == 'preflight_missing' else 'program_error'}
    if failure == 'module':
        # Source an override of the module initializer so the test is host-independent.
        module_init = tmp_path / 'modules.sh'
        module_init.write_text('module() { return 7; }\n')
        path = tmp_path / 'job/env.sh'
        path.write_text(path.read_text().replace('/etc/profile.d/modules.sh', str(module_init)))
        env['MDGEN_MODULES'] = 'test-module'
    if failure == 'missing_venv':
        env.pop('MDGEN_VENV')
    if failure == 'wrong_root':
        (tmp_path / 'mdgen/wrapper.py').unlink()
    result = subprocess.run(['bash', 'job/submit_training_region_probe.sh', '--paired-shard'],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    expected_success = failure == 'preflight_missing'
    assert (result.returncode == 0) == expected_success, result.stderr
    assert (tmp_path / 'shard-started').exists() == expected_success
    assert not (tmp_path / 'activation-continued').exists()
    setup = next((tmp_path / 'results').glob('paired-setup-*'))
    assert (setup / 'setup.log').exists()
    if not expected_success:
        assert (setup / 'setup-failure.txt').read_text().strip()
