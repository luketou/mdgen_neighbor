"""#43 凍結訓練矩陣：產生分片、隔離執行個案與保留缺口的彙整。"""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

CONTRACT_DIR = Path(__file__).resolve().parents[1] / 'docs/validation/43-contract'
STRATA = ('protein', 'replica', 'frames', 't', 'noise_seed', 'halo', 'coverage', 'control_type', 'paired_common_subset', 'boundary_ring')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    """獨佔建立，重跑不可覆寫舊產物。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def make_manifest(cases_per_shard=1):
    from .full_length.group_region import candidate_seed_groups
    if cases_per_shard < 1:
        raise ValueError('cases_per_shard must be positive')
    contract = json.loads((CONTRACT_DIR / 'frozen-contract-v1.1.json').read_text())
    inventory = json.loads((CONTRACT_DIR / 'infeasible-groups.json').read_text())['proteins']
    cases = []
    for protein, replica in itertools.product(('6sms_A', '6lrd_A', '6l3r_E', '6sup_A'), (1, 2, 3)):
        info = inventory[f'{protein}_R{replica}']
        candidates = candidate_seed_groups(info['groups'], 8)
        if candidates != info['candidate_seed_groups']:
            raise ValueError('frozen candidate permutation differs')
        regions = [(1, group) for group in candidates]
        regions += [(2, 95)] if protein == '6sup_A' else [(2, group) for group in candidates]
        canonical = {}
        for frames, t, noise, (halo, group), kind in itertools.product(
                (64, 4, 250) if protein == '6sup_A' else (250, 4),
                (0.25, 0.5, 0.75), (0, 1, 2), regions, ('paired', 'same_path_repeat')):
            case = dict(protein=protein, replica=replica, frames=frames, t=t, noise_seed=noise,
                        halo=halo, group=group, control_type=kind,
                        coverage='truncated' if protein == '6sup_A' else 'full_coverage',
                        paired_common_subset=protein != '6sup_A' or group == 95,
                        boundary_ring='all_loss_residues')
            case['case_id'] = 'c-' + digest(case)[:20]
            case['plan_status'] = 'ready'
            if protein == '6sup_A' and frames == 250:
                case.update(plan_status='phase1_infeasible', failure_category='oom',
                    evidence=['docs/validation/43-49501/README.md', 'docs/validation/43-49513/README.md'])
            elif protein != '6sup_A':
                key = (frames, t, noise, halo, kind)
                if key in canonical:
                    case.update(plan_status='duplicate', duplicate_of=canonical[key])
                else:
                    canonical[key] = case['case_id']
            cases.append(case)
    ready = [case for case in cases if case['plan_status'] == 'ready']
    shards = []
    for start in range(0, len(ready), cases_per_shard):
        ids = [c['case_id'] for c in ready[start:start + cases_per_shard]]
        shards.append({'shard_id': 's-' + digest(ids)[:20], 'case_ids': ids})
    manifest = {'schema_version': 1, 'contract_version': '1.1', 'contract_sha256': digest(contract),
                'inventory_sha256': digest(inventory),
                'selection_inventory': {name: {
                    'candidates': info['candidate_seed_groups'],
                    'halo': {halo: {'feasible': entry['feasible'], 'total': entry['total'],
                        'infeasible_candidates': [v for v in entry['infeasible']
                            if v['seed_group'] in info['candidate_seed_groups']]}
                        for halo, entry in info['halo'].items()}}
                    for name, info in inventory.items()},
                'cases_per_shard': cases_per_shard, 'cases': cases, 'shards': shards}
    manifest['manifest_id'] = digest(manifest)
    return manifest


def validate_manifest(manifest):
    expected = make_manifest(manifest['cases_per_shard'])
    if manifest != expected:
        raise ValueError('manifest differs from frozen matrix; do not fill or change cases')


def failure_category(error, log=''):
    """OOM、缺 GPU、逾時與一般錯誤分開，不以 forward 代替失敗 backward。"""
    if isinstance(error, subprocess.TimeoutExpired):
        return 'timeout'
    message = str(error) + '\n' + log
    if type(error).__name__ == 'OutOfMemoryError' or 'CUDA out of memory' in message or 'OutOfMemoryError' in message:
        return 'oom'
    if any(token in message for token in ('missing_gpu', 'No CUDA GPUs', 'No devices were found',
                                          'CUDA_ERROR_NO_DEVICE', 'no NVIDIA driver',
                                          'CUDA driver initialization failed')):
        return 'missing_gpu'
    return 'program_error'


def case_command(case, root, checkpoint, output):
    """每個個案透過正式 probe CLI，固定第一幀及核准空間／數值設定。"""
    root, output = Path(root), Path(output)
    with (root / 'validation.csv').open() as stream:
        row = next(row for row in csv.DictReader(stream) if row['name'] == case['protein'])
    inventory = json.loads((CONTRACT_DIR / 'infeasible-groups.json').read_text())['proteins']
    if len(row['seqres']) != inventory[f"{case['protein']}_R{case['replica']}"]['residues']:
        raise ValueError('input sequence length differs from frozen inventory')
    split = output.parent / 'split.csv'
    with split.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    return [sys.executable, '-m', 'mdgen.train_step_probe', '--output', str(output),
        '--paired_region', '--control_type', case['control_type'],
        '--sim_ckpt', str(checkpoint), '--time', str(case['t']), '--seed', str(case['noise_seed']),
        '--seed_group', str(case['group']), '--warmup', '1', '--repeats', '3', '--',
        '--train_split', str(split), '--val_split', str(split), '--data_dir', str(root / 'arrays'),
        '--partition_dir', str(root / 'partitions'), '--suffix', '_i400ps', '--atlas',
        '--replicates', str(case['replica']), '--overfit_frame', '--batch_size', '1',
        '--crop', str(len(row['seqres'])), '--num_frames', str(case['frames']),
        '--model_dir', str(output.parent / 'model'), '--dropout', '0', '--deterministic',
        '--grad_checkpointing', '--precision', 'fp32-reference', '--local_spatial',
        '--spatial_cell_width', '16', '--spatial_contact_radius', '16', '--core_capacity', '128',
        '--ghost_capacity', '128', '--no-cell_summaries', '--group_batch_size', '4',
        '--ffn_chunk_size', '65536', '--region_residues', '896', '--halo_rings', str(case['halo'])]


def run_shard(manifest, shard_id, output_root, data_root, checkpoint, timeout=1800,
              preflight_report=None):
    """每個分片獨立輸出；每個 case 用子程序隔離 CUDA/OOM。"""
    validate_manifest(manifest)
    shard = next(s for s in manifest['shards'] if s['shard_id'] == shard_id)
    run_id = uuid.uuid4().hex
    directory = Path(output_root).resolve() / shard_id / run_id
    directory.mkdir(parents=True, exist_ok=False)
    result = {'schema_version': 1, 'manifest_id': manifest['manifest_id'], 'shard_id': shard_id,
              'run_id': run_id, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'cases': []}
    preflight_error = None
    if preflight_report is not None:
        try:
            result['preflight'] = json.loads(Path(preflight_report).read_text())
            if not result['preflight'].get('passed'):
                raise RuntimeError(result['preflight'].get('error', 'GPU preflight failed'))
        except Exception as error:
            preflight_error = error
    checkpoint_hash = None
    checkpoint_error = None
    try:
        with Path(checkpoint).open('rb') as stream:
            checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        approved = json.loads((CONTRACT_DIR / 'frozen-contract-v1.1.json').read_text())
        if checkpoint_hash != approved['fixed_settings']['checkpoint_sha256']:
            raise ValueError('checkpoint hash differs from frozen contract')
    except Exception as error:
        checkpoint_error = error
    result['checkpoint_sha256'] = checkpoint_hash
    for identity in shard['case_ids']:
        case = next(c for c in manifest['cases'] if c['case_id'] == identity)
        child = directory / identity
        child.mkdir()
        record = {'case_id': identity, 'status': 'failed', 'exit_code': None}
        try:
            if preflight_error is not None:
                record['stage'] = 'preflight'
                raise preflight_error
            if checkpoint_error is not None:
                raise checkpoint_error
            command = case_command(case, data_root, checkpoint, child / 'probe.json')
            record['command'] = command
            with (child / 'stdout.log').open('w') as stdout, (child / 'stderr.log').open('w') as stderr:
                process = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout,
                                         env={**os.environ, 'MDGEN_PAIRED_REQUIRE_GPU': '1',
                                              'MODEL_DIR': str(child / 'model')})
            record['exit_code'] = process.returncode
            if process.returncode:
                log = (child / 'stderr.log').read_text()[-16000:]
                raise RuntimeError(log)
            record['report'] = json.loads((child / 'probe.json').read_text())
            report = record['report']
            alignment = report['alignment']
            if (alignment['full_coverage_or_truncated'] != case['coverage']
                    or alignment['paired_common_subset'] != case['paired_common_subset']
                    or alignment['t'] != case['t'] or report['seed'] != case['noise_seed']):
                raise ValueError('actual paired case differs from frozen manifest')
            record['status'] = report['status']
            if record['status'] == 'failed':
                nonfinite = any(not run['loss_finite'] or run['gradient_nonfinite']
                    or run['parameters_nonfinite_after_step'] or any(run.get('nonfinite_counts', {}).values())
                    for measurement in report['measurements']
                    for run in (measurement['full'], measurement['candidate']))
                record['failure_category'] = 'nonfinite' if nonfinite else 'control_failure'
        except Exception as error:
            record.update(failure_category=failure_category(error), error=str(error)[-16000:])
            evidence = child / 'failure.json'
            if evidence.exists():
                record['evidence'] = json.loads(evidence.read_text())
                record['failure_category'] = record['evidence']['failure_category']
        record['artifact_dir'] = str(child)
        result['cases'].append(record)
        # 每個個案完成即獨立落盤，硬逾時仍可查核已完成部分。
        write_json(child / 'case-result.json', record)
    result['status'] = 'failed' if any(c['status'] == 'failed' for c in result['cases']) else 'completed'
    write_json(directory / 'shard.json', result)
    return directory / 'shard.json'


def aggregate(manifest, results):
    """不補缺口、不跨蛋白判定；重跑保留並要求明確選取，避免挑最好結果。"""
    validate_manifest(manifest)
    expected = {s['shard_id']: s for s in manifest['shards']}
    records, seen_shards = {}, set()
    for result in results:
        if result['manifest_id'] != manifest['manifest_id'] or result['shard_id'] not in expected:
            raise ValueError('foreign manifest or shard')
        if result['shard_id'] in seen_shards:
            raise ValueError('multiple runs for shard; explicitly select one run')
        seen_shards.add(result['shard_id'])
        allowed = expected[result['shard_id']]['case_ids']
        for record in result['cases']:
            if record['status'] not in ('passed', 'descriptive', 'no_supervision', 'failed'):
                raise ValueError('unknown case status')
            if record['status'] != 'failed' and 'report' not in record:
                raise ValueError('completed case has no probe report')
            if record['case_id'] not in allowed or record['case_id'] in records:
                raise ValueError('unexpected or duplicate case result')
            records[record['case_id']] = record
    strata = {}
    for case in manifest['cases']:
        key = tuple(case[k] for k in STRATA)
        group = strata.setdefault(key, {**dict(zip(STRATA, key)), 'cases': [],
            'counts': dict(attempted=0, feasible=0, skipped=0, duplicate=0, no_supervision=0, nonfinite=0),
            'status': '完成', 'failure_categories': []})
        if case['plan_status'] != 'ready':
            record = {'case_id': case['case_id'], 'status': case['plan_status'],
                      'failure_category': case.get('failure_category'),
                      'duplicate_of': case.get('duplicate_of'), 'evidence': case.get('evidence')}
            group['counts']['duplicate' if case['plan_status'] == 'duplicate' else 'skipped'] += 1
        else:
            record = records.get(case['case_id'], {'case_id': case['case_id'], 'status': 'missing',
                                                   'failure_category': 'missing_shard'})
            group['counts']['attempted'] += record['status'] != 'missing'
            group['counts']['feasible'] += record['status'] in ('passed', 'descriptive', 'no_supervision')
            group['counts']['no_supervision'] += record['status'] == 'no_supervision'
            for measurement in record.get('report', {}).get('measurements', []):
                if not measurement['warmup']:
                    group['counts']['nonfinite'] += sum(
                        run.get('gradient_nonfinite', 0) + run.get('parameters_nonfinite_after_step', 0)
                        + sum(run.get('nonfinite_counts', {}).values())
                        for run in (measurement['full'], measurement['candidate']))
            if record['status'] in ('missing', 'failed'):
                group['status'] = '未完成'
                group['failure_categories'].append(record.get('failure_category', 'numerical_failure'))
        group['cases'].append(record)
    for group in strata.values():
        unique = {}
        for record_index, record in enumerate(group['cases']):
            alignment = record.get('report', {}).get('alignment')
            if alignment is None:
                continue
            identity = tuple(alignment['loss_group_ids'])
            if identity in unique:
                # 長蛋白候選若也重複，保留原始量測但不當獨立 region。
                record = {**record, 'status': 'duplicate', 'duplicate_of': unique[identity]}
                group['counts']['duplicate'] += 1
            else:
                unique[identity] = record['case_id']
            group['cases'][record_index] = record
        group['statistics'] = summarize(group['cases'])
    return {'schema_version': 1, 'manifest_id': manifest['manifest_id'],
            'selection_inventory': manifest['selection_inventory'],
            'status': '未完成' if any(g['status'] == '未完成' for g in strata.values()) else '完成',
            'missing_shards': sorted(set(expected) - seen_shards), 'strata': list(strata.values())}


def summarize(records):
    import numpy as np
    values, sources = {}, {}
    loss_ids, region_ids, length = set(), set(), None
    for record in records:
        report = record.get('report', {})
        alignment = report.get('alignment', {})
        loss_ids.update(alignment.get('loss_residue_ids', []))
        if alignment:
            region_ids.add(tuple(alignment['loss_group_ids']))
            length = alignment['full_length']
        if record['status'] not in ('passed', 'descriptive'):
            continue
        for measurement in report.get('measurements', []):
            if measurement['warmup']:
                continue
            for side in ('full', 'candidate'):
                for metric in ('step_time_s', 'peak_memory_allocated_bytes', 'peak_memory_reserved_bytes'):
                    name = f'{side}.{metric}'
                    values.setdefault(name, []).append(measurement[side].get(metric))
                    sources.setdefault(name, set()).add(record['case_id'])
            comparison = measurement['comparison']
            for key in ('signed_difference', 'absolute_difference', 'relative_difference'):
                name = 'loss.' + key
                values.setdefault(name, []).append(comparison['loss'][key])
                if comparison['loss'][key] is not None:
                    sources.setdefault(name, set()).add(record['case_id'])
            for category in ('gradients', 'updates'):
                for name, metrics in comparison[category].items():
                    for metric, value in metrics.items():
                        if isinstance(value, (float, int)) and not isinstance(value, bool):
                            key = f'{category}.{name}.{metric}'
                            values.setdefault(key, []).append(value)
                            sources.setdefault(key, set()).add(record['case_id'])
                    for metric in ('reference_zero', 'one_sided_zero', 'near_zero_reference'):
                        key = f'{category}.{name}.{metric}'
                        values.setdefault(key, []).append(int(metrics[metric]))
                        sources.setdefault(key, set()).add(record['case_id'])
    stats = {}
    for key, samples in values.items():
        samples = [v for v in samples if v is not None]
        if samples:
            stats[key] = {'median': float(np.median(samples)), 'p95': float(np.quantile(samples, .95)),
                          'max': max(samples), 'measurement_count': len(samples),
                          'effective_samples': len(sources.get(key, set()))}
    return {'metrics': stats, 'deduplicated_regions': len(region_ids),
            'loss_residue_union_fraction': len(loss_ids) / length if length else None,
            'repeat_policy': '重複量測不是獨立空間樣本'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    generate = sub.add_parser('manifest')
    generate.add_argument('--output', required=True)
    generate.add_argument('--cases-per-shard', type=int, default=1)
    run = sub.add_parser('run')
    run.add_argument('--manifest')
    run.add_argument('--cases-per-shard', type=int, default=1)
    run.add_argument('--shard-id', required=True)
    run.add_argument('--output-root', required=True)
    run.add_argument('--data-root', required=True)
    run.add_argument('--checkpoint', required=True)
    run.add_argument('--timeout', type=int, default=1800)
    run.add_argument('--preflight-report')
    merge = sub.add_parser('aggregate')
    merge.add_argument('--manifest', required=True)
    merge.add_argument('--results', nargs='*', default=[])
    merge.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.action == 'manifest':
        write_json(args.output, make_manifest(args.cases_per_shard))
    else:
        manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else make_manifest(args.cases_per_shard)
        if args.action == 'run':
            output = run_shard(manifest, args.shard_id, args.output_root, args.data_root,
                               args.checkpoint, args.timeout, args.preflight_report)
            print(output)
            if json.loads(output.read_text())['status'] == 'failed':
                raise SystemExit(1)
        else:
            write_json(args.output, aggregate(manifest,
                [json.loads(Path(path).read_text()) for path in args.results]))


if __name__ == '__main__':
    main()
