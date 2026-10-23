"""Use real CLI processes and local stand-ins for SSH and Slurm."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'scripts/hpc.py'


@pytest.fixture
def remote(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    (repo / 'probe.py').write_text('print("fixture")\n')
    subprocess.run(['git', '-C', str(repo), 'add', 'probe.py'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Fixture', '-c',
                    'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
    config = yaml.safe_load((ROOT / 'hpc/project.yaml').read_text())
    config['remote']['runs_root'] = str(tmp_path / 'remote')
    config['ssh'].update(target='fixture', options=[], command_timeout_seconds=2)
    config['smoke']['entrypoint'] = 'probe.py'
    config['runtime'].update(python=sys.executable, env={'MDGEN_MODULES': ''})
    path = repo / 'project.yaml'
    path.write_text(yaml.safe_dump(config))
    commands = tmp_path / 'bin'
    commands.mkdir()
    ssh = commands / 'ssh'
    ssh.write_text('#!/bin/bash\nexec bash -c "${@: -1}"\n')
    ssh.chmod(0o755)
    yield repo, path, {**os.environ, 'PATH': str(commands) + ':' + os.environ['PATH']}
    # The fake SSH host shares this test's filesystem; thaw only its temporary snapshots.
    for item in tmp_path.rglob('*'):
        if not item.is_symlink():
            item.chmod(0o755 if item.is_dir() else 0o644)


def cli(remote, *args):
    repo, config, env = remote
    return subprocess.run([sys.executable, str(CLI), '--config', str(config), *args],
                          cwd=repo, env=env, capture_output=True, text=True)


def test_prepare_freezes_selected_code_and_reports_verified_snapshot(remote):
    result = cli(remote, 'prepare', '--files', 'probe.py')
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)
    assert state['status'] == 'ready'
    assert (Path(state['remote']) / 'code/probe.py').read_text() == 'print("fixture")\n'
    assert state['files']['probe.py']['sha256']
    assert state['commit']


def test_submit_failure_is_reported_as_failed_not_completed_successfully(remote):
    repo, config, env = remote
    (repo / 'probe.py').write_text('import argparse,json,pathlib\np=argparse.ArgumentParser();p.add_argument("--require-gpu",action="store_true");p.add_argument("--output");a=p.parse_args();pathlib.Path(a.output).write_text(json.dumps({"passed":False,"status":"failed"}));raise SystemExit(1)\n')
    sbatch = Path(env['PATH'].split(':')[0]) / 'sbatch'
    sbatch.write_text('#!/bin/bash\nbash "${@: -1}" >/dev/null 2>&1\necho 123\n')
    sbatch.chmod(0o755)
    queue = sbatch.with_name('squeue')
    queue.write_text('#!/bin/bash\nexit 0\n')
    queue.chmod(0o755)
    ready = json.loads(cli(remote, 'prepare', '--files', 'probe.py').stdout)
    submitted = cli(remote, 'submit', '--state', ready['state_file'])
    assert submitted.returncode == 0, submitted.stderr
    assert json.loads(submitted.stdout)['job_id'] == '123'
    result = cli(remote, 'status', '--state', ready['state_file'])
    assert result.returncode == 1
    assert json.loads(result.stdout)['status'] == 'failed'


def test_ambiguous_submission_is_not_retried(remote):
    repo, config, env = remote
    sbatch = Path(env['PATH'].split(':')[0]) / 'sbatch'
    sbatch.write_text('#!/bin/bash\necho submitted >> "' + str(repo / 'submissions') + '"\nexit 255\n')
    sbatch.chmod(0o755)
    ready = json.loads(cli(remote, 'prepare', '--files', 'probe.py').stdout)
    assert cli(remote, 'submit', '--state', ready['state_file']).returncode != 0
    assert json.loads(Path(ready['state_file']).read_text())['status'] == 'submission_unknown'
    assert cli(remote, 'submit', '--state', ready['state_file']).returncode != 0
    assert (repo / 'submissions').read_text().splitlines() == ['submitted']


def test_changed_snapshot_cannot_be_submitted(remote):
    ready = json.loads(cli(remote, 'prepare', '--files', 'probe.py').stdout)
    payload = Path(ready['remote']) / 'code/probe.py'
    payload.chmod(0o644)
    payload.write_text('changed\n')
    result = cli(remote, 'submit', '--state', ready['state_file'])
    assert result.returncode != 0
    assert json.loads(Path(ready['state_file']).read_text())['status'] == 'ready'


def test_collect_requires_both_success_exit_and_passed_report(remote):
    ready = json.loads(cli(remote, 'prepare', '--files', 'probe.py').stdout)
    state_path = Path(ready['state_file'])
    ready.update(status='submitted', job_id='123')
    state_path.write_text(json.dumps(ready))
    commands = Path(remote[2]['PATH'].split(':')[0])
    (commands / 'squeue').write_text('#!/bin/bash\nexit 0\n')
    (commands / 'squeue').chmod(0o755)
    results = Path(ready['remote']) / 'results'
    (results / 'exit-status').write_text('0\n')
    (results / 'preflight.json').write_text('{"passed": false, "status": "failed"}')
    result = cli(remote, 'collect', '--state', str(state_path))
    assert result.returncode == 1
    state = json.loads(result.stdout)
    assert state['status'] == 'failed'
    assert json.loads((Path(state['collected']) / 'results/preflight.json').read_text())['passed'] is False
    (results / 'preflight.json').write_text('{"passed": true, "status": "passed"}')
    result = cli(remote, 'collect', '--state', str(state_path))
    assert result.returncode == 0
    assert json.loads(result.stdout)['status'] == 'passed'


def test_prepare_excludes_credentials_before_upload(remote):
    (remote[0] / '.env').write_text('fixture credential')
    result = cli(remote, 'prepare', '--files', 'probe.py', '--extra-files', '.env')
    assert result.returncode != 0
    config = yaml.safe_load(remote[1].read_text())
    assert not Path(config['remote']['runs_root']).exists()


def test_snapshot_preserves_relative_symlink_without_following_it(remote):
    repo = remote[0]
    (repo / 'alias.py').symlink_to('probe.py')
    result = cli(remote, 'prepare', '--files', 'probe.py', '--extra-files', 'alias.py')
    assert result.returncode == 0, result.stderr
    state = json.loads(result.stdout)
    link = Path(state['remote']) / 'code/alias.py'
    assert link.is_symlink()
    assert os.readlink(link) == 'probe.py'


def test_submission_timeout_can_reconcile_without_resubmitting(remote):
    repo, config, env = remote
    sbatch = Path(env['PATH'].split(':')[0]) / 'sbatch'
    sbatch.write_text('#!/bin/bash\nexec sleep 3\n')
    sbatch.chmod(0o755)
    ready = json.loads(cli(remote, 'prepare', '--files', 'probe.py').stdout)
    assert cli(remote, 'submit', '--state', ready['state_file']).returncode != 0
    assert json.loads(Path(ready['state_file']).read_text())['status'] == 'submission_unknown'
    queue = sbatch.with_name('squeue')
    queue.write_text('#!/bin/bash\necho "456 RUNNING"\n')
    queue.chmod(0o755)
    result = cli(remote, 'status', '--state', ready['state_file'])
    assert json.loads(result.stdout)['job_id'] == '456'
    assert json.loads(result.stdout)['status'] == 'active'


def test_empty_queue_without_result_is_not_a_pass(remote):
    ready = json.loads(cli(remote, 'prepare', '--files', 'probe.py').stdout)
    ready.update(status='submitted', job_id='123')
    Path(ready['state_file']).write_text(json.dumps(ready))
    queue = Path(remote[2]['PATH'].split(':')[0]) / 'squeue'
    queue.write_text('#!/bin/bash\nexit 0\n')
    queue.chmod(0o755)
    result = cli(remote, 'status', '--state', ready['state_file'])
    assert result.returncode != 0
    assert json.loads(result.stdout)['status'] == 'result_unknown'


def test_collect_bounds_large_logs_and_rejects_oversized_reports(remote):
    ready = json.loads(cli(remote, 'prepare', '--files', 'probe.py').stdout)
    ready.update(status='submitted', job_id='123')
    Path(ready['state_file']).write_text(json.dumps(ready))
    queue = Path(remote[2]['PATH'].split(':')[0]) / 'squeue'
    queue.write_text('#!/bin/bash\nexit 0\n')
    queue.chmod(0o755)
    root = Path(ready['remote'])
    (root / 'results/exit-status').write_text('0\n')
    (root / 'results/preflight.json').write_text('{"passed":true,"status":"passed"}')
    # More than the 20 MiB download budget, but only the last 100 lines are requested.
    with (root / 'logs/slurm.log').open('w') as log:
        for _ in range(22):
            log.write('old line\n' * 131072)
        log.write('last line\n')
    result = cli(remote, 'collect', '--state', ready['state_file'])
    assert result.returncode == 0, result.stderr
    log = Path(json.loads(result.stdout)['collected']) / 'logs/slurm.log'
    assert len(log.read_text().splitlines()) == 100
    assert log.read_text().endswith('last line')
    (root / 'results/preflight.json').write_bytes(b'x' * (21 * 1024**2))
    assert cli(remote, 'collect', '--state', ready['state_file']).returncode != 0


def test_queued_job_refuses_payload_changed_after_submit(remote):
    repo, config, env = remote
    sbatch = Path(env['PATH'].split(':')[0]) / 'sbatch'
    queued = repo / 'queued.sh'
    sbatch.write_text('#!/bin/bash\ncp "${@: -1}" "' + str(queued) + '"\necho 123\n')
    sbatch.chmod(0o755)
    ready = json.loads(cli(remote, 'prepare', '--files', 'probe.py').stdout)
    assert cli(remote, 'submit', '--state', ready['state_file']).returncode == 0
    payload = Path(ready['remote']) / 'code/probe.py'
    payload.chmod(0o644)  # Deliberately bypass the freeze to exercise the queued-job gate.
    payload.write_text('from pathlib import Path\nPath("' + str(repo / 'executed') + '").touch()\n')
    job = subprocess.run(['bash', str(queued)], cwd=repo, env=env, capture_output=True)
    assert job.returncode != 0
    assert not (repo / 'executed').exists()
    assert (Path(ready['remote']) / 'results/exit-status').read_text().strip() != '0'


def test_configured_cuda_home_reaches_the_actual_job(remote):
    repo, config, env = remote
    settings = yaml.safe_load(config.read_text())
    settings['runtime']['env']['CUDA_HOME'] = '/fixture/cuda-12.8'
    config.write_text(yaml.safe_dump(settings))
    (repo / 'probe.py').write_text(
        'import argparse,json,os,pathlib\n'
        'p=argparse.ArgumentParser();p.add_argument("--require-gpu",action="store_true");'
        'p.add_argument("--output");a=p.parse_args()\n'
        'passed=os.environ.get("CUDA_HOME")=="/fixture/cuda-12.8"\n'
        'pathlib.Path(a.output).write_text(json.dumps({"passed":passed}))\n'
        'raise SystemExit(0 if passed else 1)\n')
    commands = Path(env['PATH'].split(':')[0])
    (commands / 'sbatch').write_text('#!/bin/bash\nbash "${@: -1}" >/dev/null 2>&1\necho 123\n')
    (commands / 'squeue').write_text('#!/bin/bash\nexit 0\n')
    for name in ('sbatch', 'squeue'):
        (commands / name).chmod(0o755)
    ready = json.loads(cli(remote, 'prepare', '--files', 'probe.py').stdout)
    assert cli(remote, 'submit', '--state', ready['state_file']).returncode == 0
    result = cli(remote, 'collect', '--state', ready['state_file'])
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['status'] == 'passed'
