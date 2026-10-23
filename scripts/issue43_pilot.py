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
LOCAL = ['--local_spatial', '--spatial_cell_width', '16', '--spatial_contact_radius', '16',
         '--core_capacity', '128', '--ghost_capacity', '128', '--no-cell_summaries',
         '--group_batch_size', '4', '--ffn_chunk_size', '65536']
CHECKPOINT_SHA256 = 'bc90c66e7539f52c9d0514aeddd96c552061150a1eef3b10c120bc3aaff08015'


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
                                torch.Generator().manual_seed(43002))[:4].tolist()
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
    parser.add_argument('--mode', choices=('inventory', 'forward', 'train'), required=True)
    parser.add_argument('--frames', type=int, choices=(4, 250), required=True)
    parser.add_argument('--data_root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open('rb') as stream:
        checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError('checkpoint hash differs from approved Ticket 1 checkpoint')
    root = args.data_root
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
            '--times', '0.5', *LOCAL]
    cases = []
    if args.mode == 'inventory':
        for replica in (1, 2, 3):
            cases.append({'id': f'inventory-R{replica}', 'command': [*halo,
                '--replicate', str(replica), '--regions', '8', '--halo_rings', '1', '2',
                '--inventory_only']})
    else:
        # Visit all proteins in R1 before R2/R3. Order is fixed before seeing errors.
        for replica in (1, 2, 3):
            for protein, length in zip(PROTEINS, LENGTHS):
                settings = ([{'setting': 'paired'}, {'setting': 'halo1'}]
                            if args.mode == 'forward' else
                            [{'setting': 'full'}, *training_candidates(root, protein, replica)])
                for candidate in settings:
                    setting = candidate['setting']
                    identity = f'{protein}-R{replica}-{setting}'
                    if 'seed_group' in candidate:
                        identity += f"-group{candidate['seed_group']}"
                    if args.mode == 'forward':
                        command = [*halo, '--pdb_id', protein, '--replicate', str(replica),
                                   '--regions', '4', '--warmup', '1', '--repeats', '3',
                                   '--capacity_only',
                                   '--halo_rings', *(['1', '2'] if setting == 'paired' else ['1'])]
                    else:
                        # A one-row split fixes protein selection without copying any trajectory.
                        import csv
                        with (root / 'validation.csv').open() as stream:
                            rows = list(csv.DictReader(stream))
                        selected = next(row for row in rows if row['name'] == protein)
                        split = args.output_dir / f'{protein}.csv'
                        with split.open('w', newline='') as stream:
                            writer = csv.DictWriter(stream, fieldnames=selected.keys())
                            writer.writeheader()
                            writer.writerow(selected)
                        command = [sys.executable, '-m', 'mdgen.train_step_probe',
                            '--sim_ckpt', str(args.checkpoint), '--seed', '0', '--time', '0.5',
                            *(['--seed_group', str(candidate['seed_group'])]
                              if setting != 'full' else []),
                            '--configs', *(['fp32-reference'] * 4), '--',
                            '--train_split', str(split), '--val_split', str(split),
                            '--data_dir', str(root / 'arrays'), '--suffix', '_i400ps',
                            '--partition_dir', str(root / 'partitions'), '--atlas',
                            '--replicates', str(replica), '--overfit_frame', '--batch_size', '1',
                            '--num_frames', str(args.frames), '--crop', str(length),
                            '--precision', 'fp32-reference', '--grad_checkpointing',
                            '--dropout', '0', '--deterministic', *LOCAL]
                        if setting != 'full':
                            command.extend(['--training_region', '--region_residues', '896',
                                            '--halo_rings', setting[-1]])
                    cases.append({'id': identity, 'protein': protein, 'replica': replica,
                                  **candidate, 'command': command})
    report = {'mode': args.mode, 'frames': args.frames, 'checkpoint_sha256': checkpoint_hash,
              'input_manifest': input_manifest,
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'warmup': 1, 'measured_repeats': 3,
              'training_comparison': 'capacity only; full and region have different supervision',
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
        case['status'] = case.get('pre_status', 'not_run_budget')
        output = args.output_dir / f"{case['id']}.json"
        # Probe arguments must precede the separator for train.py arguments.
        command = case['command']
        where = command.index('--') if '--' in command else len(command)
        command[where:where] = ['--output', str(output)]
        case['output'] = str(output)
    save()
    start = time.monotonic()
    for case in cases:
        if case['status'] in ('duplicate', 'infeasible_seed'):
            continue
        remaining = 27 * 60 - (time.monotonic() - start)
        if remaining < 30:
            break
        log = args.output_dir / f"{case['id']}.log"
        case['log'] = str(log)
        case['status'] = 'running'
        save()
        before = time.monotonic()
        with log.open('w') as stream:
            try:
                result = subprocess.run(case['command'], stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=min(remaining, 360 if args.frames == 250 else 120))
                case['returncode'] = result.returncode
                case['status'] = 'completed' if result.returncode == 0 else 'failed'
            except subprocess.TimeoutExpired:
                case['status'] = 'timeout'
        case['elapsed_s'] = time.monotonic() - before
        text = log.read_text(errors='replace')
        if case['status'] == 'failed':
            case['failure_class'] = ('oom' if 'out of memory' in text.lower() else
                'infeasible_seed' if 'cannot hold seed group' in text else 'program_error')
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
    return report


if __name__ == '__main__':
    main()
