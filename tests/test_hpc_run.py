"""Run the remote script `hpc_run.py` builds under local bash, with a stand-in for sbatch."""
import importlib.util
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('hpc_run', ROOT / 'scripts/hpc_run.py')
hpc_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hpc_run)


@pytest.mark.parametrize('path', ['../x', 'results/../../x', '/etc/x', '//server/share/x',
                                  'C:/x', 'results\\x', ''])
def test_fetch_rejects_paths_outside_the_run_directory(path):
    with pytest.raises(SystemExit, match='relative path inside the run directory'):
        hpc_run.check_fetch_paths([path])


def test_fetch_accepts_run_relative_paths():
    hpc_run.check_fetch_paths(['results/halo_error_49339/halo_error.json', 'logs/49339.out'])


def test_record_appends_submissions_including_unknown_ones(tmp_path):
    path = tmp_path / 'hpc_job_history'
    hpc_run.record(path, '4242', 'run-1', 'run-1:smoke:120000', ['job/a.sh', '-k', 'a and b'])
    hpc_run.record(path, 'submission_unknown', 'run-1', 'run-1:smoke:120100', ['job/a.sh'])

    first, second = path.read_text(encoding='utf-8').splitlines()
    assert first.split(' ', 1)[1] == "4242 run-1 run-1:smoke:120000 job/a.sh -k 'a and b'"
    assert second.split(' ', 1)[1] == 'submission_unknown run-1 run-1:smoke:120100 job/a.sh'


@pytest.mark.skipif(shutil.which('bash') is None, reason='needs bash')
def test_submit_records_arguments_literally(tmp_path):
    run = tmp_path / 'run'
    (run / 'src').mkdir(parents=True)
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    sbatch = bin_dir / 'sbatch'
    sbatch.write_text('#!/bin/sh\necho "4242;cluster"\n')
    sbatch.chmod(0o755)
    marker = tmp_path / 'expanded'
    argv = ['job/submit_pytest.sh', '-k', 'a and b', f'$(touch {marker})', '${X:?boom}']
    comment = 'run-1:smoke:120000'

    script = hpc_run.submit_script(str(run), comment, ['K=v w'], ['--time=00:05:00'], argv)
    r = subprocess.run(['bash', '-s'], input=script, capture_output=True, text=True,
                       env={'PATH': f'{bin_dir}:/usr/bin:/bin'})

    assert r.returncode == 0, r.stderr
    assert 'JOB=4242' in r.stdout
    assert not marker.exists()
    assert (run / 'job_history').read_text() == f'4242 {comment} {shlex.join(argv)}\n'
