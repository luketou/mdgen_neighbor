"""`scripts/check_repository.py` launcher rule: CPU partitions opt out of the GPU preflight."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('check_repository', ROOT / 'scripts/check_repository.py')
check_repository = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_repository)


def launcher(partition, opt_out):
    export = 'export MDGEN_REQUIRE_GPU=0\n' if opt_out else ''
    return f'#!/bin/bash\n#SBATCH --partition={partition}\n{export}source job/env.sh\n'


@pytest.mark.parametrize('partition, opt_out, message', [
    ('cpu-2g', False, 'needs MDGEN_REQUIRE_GPU=0'),
    ('l40s', True, 'skips the GPU preflight'),
])
def test_launcher_preflight_must_match_partition(tmp_path, partition, opt_out, message):
    (tmp_path / 'submit_x.sh').write_text(launcher(partition, opt_out))
    [error] = check_repository.launcher_errors(tmp_path)
    assert error.startswith('submit_x.sh:') and message in error


@pytest.mark.parametrize('partition, opt_out', [('cpu-2g', True), ('l40s', False)])
def test_launcher_preflight_accepts_matching_partition(tmp_path, partition, opt_out):
    (tmp_path / 'submit_x.sh').write_text(launcher(partition, opt_out))
    assert check_repository.launcher_errors(tmp_path) == []


def test_repository_launchers_match_their_partition():
    assert check_repository.launcher_errors(ROOT / 'job') == []
