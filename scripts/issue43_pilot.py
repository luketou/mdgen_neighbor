"""Bounded Issue #43 capacity pilot through the two production diagnostic CLIs.

Run only in a Compute Job. Each case has its own process so an OOM cannot leave
CUDA allocations in the next case. The manifest records every planned case,
including cases not reached before the job budget expires.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROTEINS = ('6sms_A', '6lrd_A', '6l3r_E', '6sup_A')
LENGTHS = (724, 705, 669, 2128)
LOCAL_SPATIAL = ['--local_spatial', '--spatial_cell_width', '16', '--spatial_contact_radius', '16',
                 '--core_capacity', '128', '--ghost_capacity', '128', '--no-cell_summaries']
CHECKPOINT_SHA256 = 'bc90c66e7539f52c9d0514aeddd96c552061150a1eef3b10c120bc3aaff08015'
DEFAULT_MEMORY = {'group_batch_size': 4, 'ffn_chunk_size': 65536, 'expandable_segments': False}
# A full-protein 2128-residue step with reduced group batching is far slower than the 4/250
# frame capacity cases, so sweep cases get a longer default timeout.
SWEEP_CASE_TIMEOUT_S = 900


def parse_memory(text):
    """`gbs=N,ffn=N,expandable` -> memory settings; omitted keys keep the approved defaults."""
    memory = dict(DEFAULT_MEMORY)
    for item in filter(None, text.split(',')):
        key, _, value = item.partition('=')
        if key == 'expandable' and not value:
            memory['expandable_segments'] = True
        elif key in ('gbs', 'ffn') and value.isdigit() and int(value) > 0:
            memory['group_batch_size' if key == 'gbs' else 'ffn_chunk_size'] = int(value)
        else:
            raise ValueError(f'invalid memory setting {item!r}')
    return memory


def parse_step(text):
    """`FRAMES[:MEMORY]` -> one sweep step."""
    frames, colon, memory = text.partition(':')
    if not frames.isdigit() or int(frames) <= 0 or (colon and not memory):
        raise ValueError(f'invalid sweep step {text!r}')
    return {'frames': int(frames), **parse_memory(memory)}


def memory_of(settings):
    """The memory settings of a memory dict or a sweep step, without its frame count."""
    return {k: settings[k] for k in DEFAULT_MEMORY}


def local_args(memory):
    return [*LOCAL_SPATIAL, '--group_batch_size', str(memory['group_batch_size']),
            '--ffn_chunk_size', str(memory['ffn_chunk_size'])]


LOCAL = local_args(DEFAULT_MEMORY)


def child_env(memory, base):
    env = dict(base)
    if memory['expandable_segments']:
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    return env


def memory_tag(memory):
    if memory_of(memory) == DEFAULT_MEMORY:
        return ''
    return (f"-gbs{memory['group_batch_size']}-ffn{memory['ffn_chunk_size']}"
            + ('-expandable' if memory['expandable_segments'] else ''))


def skip_reason(step, finished):
    """Sweep stop rules: frames beyond an OOM with the same memory settings cannot fit, and
    once a frame count fits, more aggressive memory settings for it are unnecessary."""
    for case in finished:
        earlier = case.get('step')
        if earlier is None:
            continue
        if (memory_of(earlier) == memory_of(step) and earlier['frames'] <= step['frames']
                and case['status'] == 'failed' and case.get('failure_class') == 'oom'):
            return 'more_frames_than_oom_step'
        if earlier['frames'] == step['frames'] and case['status'] == 'completed':
            return 'frames_already_fit'
    return None


def check_lengths(root):
    """Full-protein steps crop to LENGTHS, so the split's sequences must have exactly those lengths."""
    import csv
    with (root / 'validation.csv').open() as stream:
        lengths = {row['name']: len(row['seqres']) for row in csv.DictReader(stream)}
    for protein, length in zip(PROTEINS, LENGTHS):
        if lengths.get(protein) != length:
            raise ValueError(f'{protein} has {lengths.get(protein)} residues in validation.csv; '
                             f'the pilot crops it to {length}')


def select_cases(cases, ids):
    if ids is None:
        return cases
    unknown = set(ids) - {case['id'] for case in cases}
    if unknown:
        raise ValueError(f'unknown case id(s): {sorted(unknown)}')
    return [case for case in cases if case['id'] in ids]


def train_command(root, checkpoint, output_dir, identity, protein, replica, frames, setting,
                  seed_group, memory):
    import csv
    # A one-row split fixes protein selection without copying any trajectory.
    with (root / 'validation.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    selected = next(row for row in rows if row['name'] == protein)
    split = output_dir / f'{protein}.csv'
    with split.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=selected.keys())
        writer.writeheader()
        writer.writerow(selected)
    command = [sys.executable, '-m', 'mdgen.train_step_probe',
        '--sim_ckpt', str(checkpoint), '--seed', '0', '--time', '0.5',
        *(['--coverage_control'] if setting == 'coverage_control' else []),
        *(['--seed_group', str(seed_group)] if setting != 'full' else []),
        '--configs', *(['fp32-reference'] * (1 if setting == 'coverage_control' else 4)), '--',
        '--train_split', str(split), '--val_split', str(split),
        '--model_dir', str(output_dir / 'workdir' / identity),
        '--data_dir', str(root / 'arrays'), '--suffix', '_i400ps',
        '--partition_dir', str(root / 'partitions'), '--atlas',
        '--replicates', str(replica), '--overfit_frame', '--batch_size', '1',
        '--num_frames', str(frames), '--crop', str(LENGTHS[PROTEINS.index(protein)]),
        '--precision', 'fp32-reference', '--grad_checkpointing',
        '--dropout', '0', '--deterministic', *local_args(memory)]
    if setting == 'coverage_control':
        command.extend(['--region_residues', '896', '--halo_rings', '1'])
    elif setting != 'full':
        command.extend(['--training_region', '--region_residues', '896',
                        '--halo_rings', setting[-1]])
    return command


def sweep_cases(root, checkpoint, output_dir, protein, replica, steps):
    """Full-protein training steps over explicit frame counts and memory settings, in order."""
    cases = []
    for step in steps:
        identity = f"{protein}-R{replica}-full-f{step['frames']}{memory_tag(step)}"
        command = train_command(root, checkpoint, output_dir, identity, protein, replica,
                                step['frames'], 'full', None, step)
        cases.append({'id': identity, 'protein': protein, 'replica': replica, 'setting': 'full',
                      'frames': step['frames'], 'step': step, 'command': command})
    return cases


def training_candidates(root, protein, replica):
    import numpy as np
    import torch
    from mdgen.full_length.group_region import sample_region_groups
    from mdgen.full_length.partition import PartitionParams
    from mdgen.full_length.partition_store import load_partition
    from mdgen.topology import load_chain_numbering
    stem = f'{protein}_R{replica}_i400ps'
    arr = np.load(root / 'arrays' / f'{stem}.npy', mmap_mode='r')
    chain = load_chain_numbering(root / 'arrays' / f'{stem}_chain.npz', arr.shape[1])
    ca = np.asarray(arr[0, :, 1], dtype=np.float32)
    partition = load_partition(root / 'partitions' / f'{stem}_partition.npz',
                               PartitionParams(16., 16., 128, 128), ca=ca, **chain)
    candidates = torch.randperm(len(partition.groups), generator=
                                torch.Generator().manual_seed(43002))[:8].tolist()
    result = []
    for k in (1, 2):
        seen = {}
        for seed_group in candidates:
            entry = {'setting': f'halo{k}', 'seed_group': seed_group}
            try:
                selection = sample_region_groups(partition, ca, region_residues=896,
                                                 halo_rings=k, seed_group=seed_group)
                entry.update({'loss_group_ids': list(selection.loss_groups),
                              'region_residues': len(selection.residues),
                              'truncated': len(selection.residues) < len(ca)})
                if selection.loss_groups in seen:
                    entry.update({'pre_status': 'duplicate',
                                  'duplicate_of_seed_group': seen[selection.loss_groups]})
                seen.setdefault(selection.loss_groups, seed_group)
            except ValueError as error:
                entry.update({'pre_status': 'infeasible_seed', 'reason': str(error)})
            result.append(entry)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('inventory', 'forward', 'train', 'sweep'), required=True)
    parser.add_argument('--frames', type=int, help='frame count; required except in sweep mode')
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--plan_only', action='store_true',
                        help='validate inputs and write the case manifest on a CPU Compute Job')
    parser.add_argument('--memory', type=parse_memory, default=parse_memory(''),
                        help='gbs=N,ffn=N,expandable; defaults are the approved local settings')
    parser.add_argument('--cases', nargs='+', help='run only these planned case ids')
    parser.add_argument('--protein', choices=PROTEINS, help='sweep mode: protein')
    parser.add_argument('--replica', type=int, choices=(1, 2, 3), help='sweep mode: replica')
    parser.add_argument('--step', type=parse_step, action='append',
                        help='sweep mode: FRAMES[:MEMORY], run in the given order')
    parser.add_argument('--budget_minutes', type=float, default=27,
                        help='stop starting cases after this wall time')
    parser.add_argument('--case_timeout', type=float,
                        help=f'seconds per case; default {SWEEP_CASE_TIMEOUT_S} in sweep mode, '
                             'otherwise 120 for <=4 frames, else 360')
    args = parser.parse_args(argv)
    if args.mode == 'sweep':
        if args.protein is None or args.replica is None or not args.step:
            parser.error('sweep mode requires --protein, --replica and --step')
    elif args.frames is None or args.frames <= 0:
        parser.error('--frames must be a positive integer')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open('rb') as stream:
        checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError('checkpoint hash differs from approved Ticket 1 checkpoint')
    root = args.data_root
    check_lengths(root)
    inputs = [root / 'validation.csv']
    for replica in (1, 2, 3):
        for protein in PROTEINS:
            stem = f'{protein}_R{replica}_i400ps'
            inputs.extend([root / 'arrays' / f'{stem}.npy',
                           root / 'arrays' / f'{stem}_chain.npz',
                           root / 'partitions' / f'{stem}_partition.npz'])
    input_manifest = []
    for path in inputs:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        input_manifest.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest})
    halo = [sys.executable, '-m', 'mdgen.full_length.halo_error',
            '--sim_ckpt', str(args.checkpoint), '--data_dir', str(root / 'arrays'),
            '--split', str(root / 'validation.csv'), '--suffix', '_i400ps',
            '--partition_dir', str(root / 'partitions'), '--region_residues', '896',
            '--num_frames', str(args.frames), '--selection_seed', '43002', '--seed', '0',
            '--times', '0.5', *local_args(args.memory)]
    cases = []
    if args.mode == 'sweep':
        cases = sweep_cases(root, args.checkpoint, args.output_dir, args.protein, args.replica,
                            args.step)
    elif args.mode == 'inventory':
        for replica in (1, 2, 3):
            cases.append({'id': f'inventory-R{replica}', 'command': [*halo,
                '--replicate', str(replica), '--regions', '8', '--halo_rings', '1', '2',
                '--inventory_only']})
    else:
        # Revision 2: long protein R1, baselines R1, then R2/R3. Never sort by errors.
        for replica in (1, 2, 3):
            ordered = sorted(zip(PROTEINS, LENGTHS), key=lambda p: p[0] != '6sup_A')
            for protein, length in ordered:
                settings = ([{'setting': 'paired'}, {'setting': 'halo1'}]
                            if args.mode == 'forward' else
                            [{'setting': 'full'}, *training_candidates(root, protein, replica)])
                if args.mode == 'train' and protein != '6sup_A' and replica == 1:
                    settings.insert(1, {'setting': 'coverage_control',
                                        'seed_group': settings[1]['seed_group']})
                for candidate in settings:
                    setting = candidate['setting']
                    identity = f'{protein}-R{replica}-{setting}'
                    if 'seed_group' in candidate:
                        identity += f"-group{candidate['seed_group']}"
                    if args.mode == 'forward':
                        command = [*halo, '--pdb_id', protein, '--replicate', str(replica),
                                   '--regions', '8', '--warmup', '1', '--repeats', '3',
                                   '--capacity_only',
                                   '--halo_rings', *(['1', '2'] if setting == 'paired' else ['1'])]
                    else:
                        command = train_command(root, args.checkpoint, args.output_dir, identity,
                                                protein, replica, args.frames, setting,
                                                candidate.get('seed_group'), args.memory)
                    cases.append({'id': identity, 'protein': protein, 'replica': replica,
                                  **candidate, 'command': command})
    cases = select_cases(cases, args.cases)
    for case in cases:
        case.setdefault('frames', args.frames)
        case['memory'] = memory_of(case.get('step', args.memory))
        case['timeout_s'] = args.case_timeout or (
            SWEEP_CASE_TIMEOUT_S if args.mode == 'sweep' else 120 if case['frames'] <= 4 else 360)
    report = {'mode': args.mode, 'frames': args.frames, 'checkpoint_sha256': checkpoint_hash,
              # Sweep steps carry their own settings; each case records its actual `memory`.
              'memory_settings': None if args.mode == 'sweep' else args.memory,
              'selected_cases': args.cases,
              'budget_minutes': args.budget_minutes,
              'sweep_rules': (['skip larger frames after an OOM with the same memory settings',
                               'skip further memory settings once a frame count fits']
                              if args.mode == 'sweep' else None),
              'plan_only': args.plan_only, 'driver_finished': False,
              'policy_revision': 2, 'pilot_candidates': 8,
              'case_order': 'long protein R1, baselines R1, then R2/R3',
              'input_manifest': input_manifest,
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'warmup': 1, 'measured_repeats': 3,
              'training_comparison': 'capacity cases have different supervision; '
                                     'coverage controls align noise and supervision, no thresholds',
              'memory_policy': 'standalone full and region forward; reference output released',
              'cases': cases}
    if os.environ.get('MDGEN_RUN_DIR'):
        run = Path(os.environ['MDGEN_RUN_DIR'])
        report['run'] = json.loads((run / 'meta.json').read_text())
        report['snapshot_manifest_sha256'] = hashlib.sha256(
            (run / 'src' / 'manifest.sha256').read_bytes()).hexdigest()
    summary = {'mode': args.mode, 'frames': args.frames, 'reports': {}}
    manifest = args.output_dir / 'manifest.json'

    def save():
        manifest.write_text(json.dumps(report, indent=2))

    for case in cases:
        case['status'] = case.get('pre_status', 'planned_only' if args.plan_only else 'not_run_budget')
        output = args.output_dir / f"{case['id']}.json"
        # Probe arguments must precede the separator for train.py arguments.
        command = case['command']
        where = command.index('--') if '--' in command else len(command)
        command[where:where] = ['--output', str(output)]
        case['output'] = str(output)
    save()
    if args.plan_only:
        report['driver_finished'] = True
        save()
        print(f'planned {len(cases)} cases; no model executions', flush=True)
        return report
    start = time.monotonic()
    for index, case in enumerate(cases):
        if case['status'] in ('duplicate', 'infeasible_seed'):
            continue
        remaining = args.budget_minutes * 60 - (time.monotonic() - start)
        if remaining < 30:
            break
        reason = skip_reason(case['step'], cases[:index]) if 'step' in case else None
        if reason:
            case['status'] = 'skipped_by_rule'
            case['reason'] = reason
            save()
            print(case['id'], case['status'], reason, flush=True)
            continue
        log = args.output_dir / f"{case['id']}.log"
        case['log'] = str(log)
        case['status'] = 'running'
        save()
        before = time.monotonic()
        with log.open('w') as stream:
            try:
                result = subprocess.run(case['command'], stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=min(remaining, case['timeout_s']),
                                        env=child_env(case['memory'], os.environ))
                case['returncode'] = result.returncode
                case['status'] = 'completed' if result.returncode == 0 else 'failed'
            except subprocess.TimeoutExpired:
                case['status'] = 'timeout'
        case['elapsed_s'] = time.monotonic() - before
        text = log.read_text(errors='replace')
        if case['status'] == 'failed':
            case['failure_class'] = ('oom' if 'out of memory' in text.lower() else
                'infeasible_seed' if 'cannot hold seed group' in text else 'program_error')
            case['failure_phase'] = ('backward' if 'backward(' in text or 'run_backward' in text
                else 'optimizer' if 'optimizer.step' in text else 'forward_or_setup')
        if case['status'] == 'completed' and not Path(case['output']).is_file():
            case['status'] = 'missing_report'
        elif case['status'] == 'completed':
            payload = Path(case['output']).read_bytes()
            child = json.loads(payload)
            case['output_sha256'] = hashlib.sha256(payload).hexdigest()
            # Raw residue observations stay at the recorded output path on the server.
            for protein in child.get('proteins', []):
                protein.pop('observations', None)
            summary['reports'][case['id']] = child
            (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
        save()
        print(case['id'], case['status'], flush=True)
    report['driver_finished'] = True
    save()
    print('pilot manifest finalized', flush=True)
    return report


if __name__ == '__main__':
    main()
