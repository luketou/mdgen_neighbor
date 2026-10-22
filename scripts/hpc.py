"""Deploy and run the configured GPU preflight in an immutable Slurm snapshot."""
import argparse
import base64
from datetime import datetime, timezone
import fnmatch
import hashlib
import io
import json
import os
import posixpath
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import tarfile
import uuid

import yaml


def save(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2) + '\n')
    temp.replace(path)


def ssh(config, command, payload=None):
    settings = config['ssh']
    options = [item for option in settings['options'] for item in ('-o', option)]
    return subprocess.run(['ssh', *options, settings['target'], command], input=payload,
                          capture_output=True, check=True,
                          timeout=settings['command_timeout_seconds']).stdout


def git(*args):
    return subprocess.check_output(['git', *args]).decode().strip()


def allowed(name, config):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or str(path) != name:
        return False
    # Check both root-level and nested files against the configured patterns.
    return not any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch('root/' + name, p)
                   for p in config['sync']['exclude'])


def prepare(args, config):
    tracked = set(git('ls-files', '-z').split('\0'))
    names = list(dict.fromkeys(args.files + args.extra_files))
    if not names:
        raise ValueError('Select files explicitly')
    entries = {}
    for name in names:
        if not allowed(name, config):
            raise ValueError('Excluded or unsafe payload path: ' + name)
        if name not in tracked and name not in args.extra_files:
            raise ValueError('Untracked file requires --extra-files: ' + name)
        path = Path(name)
        link = os.readlink(path) if path.is_symlink() else None
        if link is not None:
            target = posixpath.normpath(str(PurePosixPath(name).parent / link))
            if PurePosixPath(link).is_absolute() or target not in names:
                raise ValueError('Symlink must point to another selected file: ' + name)
        # A directory symlink must not allow data outside the checkout into a snapshot.
        if not path.resolve().is_relative_to(Path.cwd().resolve()):
            raise ValueError('Payload resolves outside checkout: ' + name)
        if not path.is_file() or path.stat().st_size > config['sync']['max_file_mib'] * 1024**2:
            raise ValueError('Missing or oversized payload: ' + name)
        data = link.encode() if link is not None else path.read_bytes()
        entries[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data), 'link': link}
    if config['smoke']['entrypoint'] not in names or '_hpc_run.sh' in names:
        raise ValueError('Select the smoke entrypoint and reserve _hpc_run.sh for the launcher')
    run_id = 'preflight-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8]
    remote = str(PurePosixPath(config['remote']['runs_root']) / run_id)
    state_path = Path(args.state_dir).resolve() / run_id / 'manifest.json'
    state = {'run_id': run_id, 'remote': remote, 'state_file': str(state_path),
             'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain')),
             'extra_files': args.extra_files, 'files': entries, 'status': 'preparing'}
    state['config'] = config
    save(state_path, state)
    # Archive the exact bytes hashed above, rather than allowing tar to follow links.
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode='w') as tar:
        for name, entry in entries.items():
            data = os.readlink(name).encode() if entry.get('link') is not None else Path(name).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError('Payload changed during snapshot: ' + name)
            info = tarfile.TarInfo(name)
            info.mode = 0o644
            if entry.get('link') is not None:
                info.type = tarfile.SYMTYPE
                info.linkname = entry['link']
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        data = launcher(config, remote, entries).encode()
        entries['_hpc_run.sh'] = {'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data)}
        info = tarfile.TarInfo('_hpc_run.sh')
        info.size = len(data)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))
    q = shlex.quote
    ssh(config, 'mkdir -p ' + q(str(PurePosixPath(remote).parent)) + ' && mkdir ' + q(remote)
        + ' && mkdir ' + q(remote + '/code') + ' ' + q(remote + '/logs') + ' ' + q(remote + '/results'))
    ssh(config, 'tar -xf - -C ' + q(remote + '/code'), archive.getvalue())
    verify(state, config)
    freeze = ("import pathlib,sys;root=pathlib.Path(sys.argv[1]);"
              "[(p.chmod(p.stat().st_mode & ~0o222)) for p in root.rglob('*') if not p.is_symlink()];"
              "root.chmod(root.stat().st_mode & ~0o222)")
    ssh(config, shlex.join(['python3', '-c', freeze, remote + '/code']))
    state['status'] = 'ready'
    save(state_path, state)
    return state


VERIFY_PROGRAM = r"""
import hashlib,json,pathlib,os,sys
root=pathlib.Path(sys.argv[1]); expected=json.loads(sys.argv[2])
actual={n:hashlib.sha256(os.readlink(root/n).encode() if (root/n).is_symlink() else (root/n).read_bytes()).hexdigest() for n in expected}
if actual != expected: raise SystemExit('Remote snapshot hash mismatch')
"""


def verification_command(remote, entries):
    expected = {name: entry['sha256'] for name, entry in entries.items()}
    return shlex.join(['python3', '-c', VERIFY_PROGRAM, remote + '/code', json.dumps(expected)])


def verify(state, config):
    ssh(config, verification_command(state['remote'], state['files']))



def launcher(config, remote, entries):
    q = shlex.quote
    modules = shlex.split(config['runtime']['env'].get('MDGEN_MODULES', ''))
    setup = 'source /etc/profile.d/modules.sh\nmodule load ' + shlex.join(modules) if modules else ':'
    command = shlex.join([config['runtime']['python'], config['smoke']['entrypoint'],
                          '--require-gpu', '--output', remote + '/results/preflight.json'])
    return ('#!/bin/bash\nset -euo pipefail\n'
            + 'run_results=' + q(remote + '/results') + '\n'
            + 'finish() { rc=$?; printf "%s\\n" "$rc" > "$run_results/exit-status.tmp"; '
            + 'mv "$run_results/exit-status.tmp" "$run_results/exit-status"; }\n'
            + 'trap finish EXIT\ncd ' + q(remote + '/code') + '\n' + verification_command(remote, entries) + '\n' + setup + '\n' + command + '\n')



def submit(state, config):
    if state['status'] != 'ready':
        raise ValueError('Only a ready snapshot can be submitted; reconcile unknown submissions first')
    verify(state, config)
    resources = config['smoke']['resources']
    flags = {'partition': resources['partition'], 'account': resources['account'],
             'cpus-per-task': resources['cpus'], 'gres': 'gpu:' + str(resources['gpus']),
             'mem': resources['memory'], 'time': resources['time'],
             'chdir': state['remote'] + '/code', 'job-name': state['run_id'],
             'output': state['remote'] + '/logs/slurm.log'}
    if any(v is None for v in flags.values()):
        raise ValueError('Smoke resources must be configured')
    command = ['sbatch', '--parsable', *['--' + k + '=' + str(v) for k, v in flags.items()],
               state['remote'] + '/code/_hpc_run.sh']
    state.update(status='submission_unknown', submit_time=datetime.now(timezone.utc).isoformat())
    save(Path(state['state_file']), state)
    reply = ssh(config, shlex.join(command)).decode().strip().split(';')[0]
    if not reply.isdigit():
        raise ValueError('Submission reply missing job ID; reconcile by unique job name')
    state.update(status='submitted', job_id=reply)
    return state


# This read-only remote program returns only fixed result paths and a bounded log tail.
READ_RESULTS = r"""
import base64,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); limit=int(sys.argv[2]); tail=int(sys.argv[3]); files={}
for name in ['results/preflight.json','results/exit-status','logs/slurm.log']:
 p=root/name
 if p.exists():
  if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()): raise SystemExit('Unsafe result path')
  size=p.stat().st_size
  if name.endswith('.log'):
   with p.open('rb') as source:
    source.seek(max(0,size-limit)); data=source.read(limit)
   if size>limit: data=data.partition(b'\n')[2]
   data=b'\n'.join(data.splitlines()[-tail:])
  else:
   if size>limit: raise SystemExit('Result exceeds size limit')
   data=p.read_bytes()
  files[name]=base64.b64encode(data).decode()
if sum(len(base64.b64decode(v)) for v in files.values())>limit: raise SystemExit('Total result limit exceeded')
print(json.dumps(files))
"""


def status(state, config, collect=False):
    if state['status'] in ('ready', 'preparing'):
        return state
    selector = ['--jobs=' + state['job_id']] if state.get('job_id') else ['--name=' + state['run_id']]
    queue = ssh(config, shlex.join(['squeue', '--noheader', '--me',
                                  '--format=%i %T', *selector])).decode().strip()
    if queue:
        lines = queue.splitlines()
        if not state.get('job_id') and len(lines) == 1 and lines[0].split()[0].isdigit():
            state['job_id'] = lines[0].split()[0]
        state.update(status='active', queue=queue)
        return state
    limit = int(config['results']['max_download_mib'] * 1024**2)
    command = shlex.join(['python3', '-c', READ_RESULTS, state['remote'], str(limit),
                          str(config['results']['log_tail_lines'])])
    encoded = json.loads(ssh(config, command))
    files = {k: base64.b64decode(v) for k, v in encoded.items()}
    if sum(map(len, files.values())) > limit:
        raise ValueError('Total result limit exceeded')
    report = json.loads(files.get('results/preflight.json', b'{}'))
    exit_code = files.get('results/exit-status', b'').strip()
    if exit_code:
        state['status'] = 'passed' if exit_code == b'0' and report.get('passed') is True else 'failed'
    else:
        state['status'] = 'submission_unknown' if not state.get('job_id') else 'result_unknown'
    if collect:
        destination = Path(state['state_file']).parent / 'collected'
        for name, data in files.items():
            if name not in ('results/preflight.json', 'results/exit-status', 'logs/slurm.log'):
                raise ValueError('Unexpected result path')
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        state['collected'] = str(destination)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('hpc/project.yaml'))
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('prepare')
    command.add_argument('--files', nargs='+', required=True)
    command.add_argument('--extra-files', nargs='*', default=[])
    command.add_argument('--state-dir', default='.scratch/hpc')
    for name in ('submit', 'status', 'collect'):
        command = commands.add_parser(name)
        command.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    try:
        config = yaml.safe_load(args.config.read_text())
        if args.command == 'prepare':
            result = prepare(args, config)
        else:
            result = json.loads(args.state.read_text())
            if result['config'] != config:
                raise ValueError('Config changed since snapshot; use the original config')
            if args.command == 'submit':
                result = submit(result, config)
            else:
                result = status(result, config, collect=args.command == 'collect')
            save(args.state, result)
        print(json.dumps(result))
        return 1 if result['status'] in ('failed', 'result_unknown', 'submission_unknown') else 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        # SSH output can contain authentication material; keep it out of artifacts.
        print(f'HPC command failed: {type(error).__name__}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
