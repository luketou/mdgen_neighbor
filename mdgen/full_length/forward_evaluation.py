"""Evaluate raw Halo JSON against the frozen Issue #43 forward contract."""
import argparse
from collections import Counter
import hashlib
from itertools import product
import json
import math
from pathlib import Path

from .halo_error import coverage_class, observation_stats

STRATA = ('protein', 'replica', 'frames', 't', 'noise_seed', 'halo_rings', 'boundary_ring',
          'full_coverage_or_truncated', 'paired_common_subset')


def settings_mismatches(report, contract):
    fixed, seeds = contract['fixed_settings'], contract['seed_groups']
    settings = report.get('settings', {})
    frames = settings.get('num_frames') or settings.get('model', {}).get('num_frames')
    expected = {
        'checkpoint_sha256': fixed['checkpoint_sha256'],
        'settings.selection_seed': seeds['selection_seed'],
        'settings.regions': seeds['candidates_per_protein_replica'],
        'settings.region_residues': fixed['region_budget'],
        'settings.warmup': fixed['warmup'], 'settings.repeats': fixed['measured_repeats'],
        'settings.halo_rings': seeds['halo_rings'],
        'precision_state.precision_config': fixed['precision'],
        'precision_state.float32_matmul_precision': fixed['float32_matmul_precision'],
        'precision_state.cuda_matmul_allow_tf32': fixed['tf32'],
        'precision_state.cudnn_allow_tf32': fixed['tf32'],
        'settings.model.dropout': fixed['dropout'], 'settings.model.num_frames': frames}
    for key, value in fixed['local_spatial'].items():
        name = 'spatial_' + key if key in ('cell_width', 'contact_radius') else key
        expected[f'settings.model.{name}'] = value
    fields, missing = [], object()
    for path, value in expected.items():
        actual = report
        for key in path.split('.'):
            actual = actual.get(key, missing) if isinstance(actual, dict) else missing
        if actual is missing or actual != value:
            fields.append(path)
    if str(frames) not in contract['forward_matrix']['frames']:
        fields.append('settings.num_frames')
    return fields


def coverage(protein, halo, expected):
    region_ids = {rid for rid, _ in expected}
    entries = [c.get('by_halo', {}).get(str(halo), {}) for c in protein.get('candidates', [])]
    selected = [e for e in entries if e.get('region_id') in region_ids]
    skipped = sum(e.get('status') == 'infeasible' for e in entries)
    residues = {i for _, i in expected}
    return {'attempted': len(selected) + skipped, 'feasible': len(selected), 'skipped': skipped,
            'duplicate': sum(e['status'] == 'duplicate' for e in selected),
            'deduplicated_regions': len(region_ids),
            'loss_residue_union_fraction': len(residues) / protein['residues']}


def verdict(rows, thresholds, full_coverage, incomplete=False, nonfinite=False):
    if nonfinite or any(row['reference_class'] == 'nonfinite' or any(
            value is not None and not math.isfinite(value)
            for value in (row['absolute_rms'], row['relative'])) for row in rows):
        return 'fail', 'nonfinite'
    incomplete |= any((row['reference_class'] not in (
        'regular', 'zero_reference', 'near_zero_reference', 'no_supervision', 'nonfinite')) or
        (row['reference_class'] in ('regular', 'zero_reference', 'near_zero_reference') and
         row['absolute_rms'] is None) or
        (row['reference_class'] == 'regular' and row['relative'] is None) for row in rows)
    if incomplete or not rows:
        return 'incomplete', 'missing_required_result'
    summary = observation_stats(rows)
    if full_coverage:
        envelope = thresholds['full_coverage_per_residue']
        failed = any(summary[metric]['max'] is not None and
                     summary[metric]['max'] > envelope[limit]
                     for metric, limit in (('absolute_rms', 'absolute_per_sqrt_component'),
                                           ('relative', 'relative')))
    else:
        failed = any(summary[metric][stat] is not None and summary[metric][stat] > limit
                     for metric, policy in (
                         ('absolute_rms', 'absolute_l2_over_sqrt_effective_components'),
                         ('relative', 'relative'))
                     for stat, limit in thresholds['truncated'][policy].items())
    if failed:
        return 'fail', 'threshold'
    if summary['regular'] == 0:
        return 'not_evaluable', 'no_regular_samples'
    return 'pass', None


def evaluate_reports(reports, contract, ticket='3'):
    """Require every matrix cell and every residue promised by region geometry."""
    matrix = contract['forward_matrix']
    thresholds = matrix['thresholds']
    cells, report_failures = {}, []
    for report_index, report in enumerate(reports):
        fields = settings_mismatches(report, contract)
        if fields:
            report_failures.append({'report_index': report_index, 'verdict': 'fail',
                                    'reason': 'settings_mismatch',
                                    'fields': fields})
        settings = report.get('settings', {})
        for protein in report['proteins']:
            replica = settings.get('replicate')
            name = protein['name']
            suffix = f'_R{replica}'
            name = name[:-len(suffix)] if name.endswith(suffix) else name
            frames = settings.get('num_frames') or settings.get('model', {}).get('num_frames')
            for t, noise, halo in product(settings.get('times', []),
                    settings.get('noise_seeds') or [settings.get('seed', 0)], settings.get('halo_rings', [])):
                cells.setdefault((name, replica, frames, t, noise, halo), []).append((protein, fields))
    strata = []
    axes = (matrix['tickets'][str(ticket)], matrix['replicas'], map(int, matrix['frames']),
            matrix['t'], matrix['noise_seeds'], contract['seed_groups']['halo_rings'])
    for cell in product(*axes):
        base = dict(zip(STRATA[:6], cell))
        proteins = cells.get(cell, [])
        if len(proteins) != 1:
            strata.append({**base, 'boundary_ring': None, 'full_coverage_or_truncated': None,
                'paired_common_subset': None, 'verdict': 'incomplete',
                'reason': 'missing_required_result' if not proteins else 'duplicate_report',
                **observation_stats([]), 'coverage': None})
            continue
        (protein, fields), = proteins
        expected, groups = {}, {}
        for region in protein.get('regions', []):
            entry = region['by_halo'].get(str(cell[5]))
            if entry is None:
                continue
            for residue, ring in zip(entry['loss_residue_indices'], entry['boundary_rings']):
                key = (ring, coverage_class(entry['truncated']),
                       entry['paired_common_subset'])
                expected.setdefault(key, set()).add((entry['region_id'], residue))
        for row in protein.get('observations', []):
            if (row['t'], row.get('noise_seed'), row['halo_rings']) != cell[3:]:
                continue
            key = (row['boundary_ring'], coverage_class(row['truncated']),
                   row['paired_common_subset'])
            groups.setdefault(key, []).append(row)
        if not expected and not groups:
            expected[(None, None, None)] = set()
        for key in dict.fromkeys([*expected, *groups]):
            rows = groups.get(key, [])
            actual = Counter((row['region_id'], row['residue_index']) for row in rows)
            incomplete = (set(actual) != expected.get(key, set()) or
                          any(count != 1 for count in actual.values()))
            selected_ids = {c['by_halo'][str(cell[5])]['region_id']
                            for c in protein.get('candidates', [])
                            if 'region_id' in c.get('by_halo', {}).get(str(cell[5]), {})}
            expected_ids = {rid for entries in expected.values() for rid, _ in entries}
            incomplete |= selected_ids != expected_ids
            if 'groups' in protein:
                required = min(protein['groups'],
                               contract['seed_groups']['candidates_per_protein_replica'])
                incomplete |= len(protein.get('candidates', [])) != required
            ids = {rid for rid, _ in expected.get(key, set())}
            nonfinite = protein.get('nonfinite_counts', {}).get('input', 0)
            measurements = protein.get('measurements')
            if measurements is None:
                nonfinite += sum(protein.get('nonfinite_counts', {}).values())
            else:
                for path in ('full', 'regions'):
                    nonfinite += sum(m.get('nonfinite', 0) for m in measurements[path]
                        if (m['t'], m.get('noise_seed')) == cell[3:5] and
                        (path == 'full' or (m['halo_rings'] == cell[5] and m.get('region_id') in ids)))
            status, reason = verdict(rows, thresholds, key[1] == 'full_coverage', incomplete,
                                     nonfinite)
            candidates = protein.get('candidates', [])
            all_infeasible = bool(candidates) and all(
                c.get('by_halo', {}).get(str(cell[5]), {}).get('status') == 'infeasible'
                for c in candidates)
            if (all_infeasible and not incomplete and not rows
                    and not expected.get(key) and not nonfinite):
                status, reason = 'no_feasible_region', 'all_candidates_infeasible'
            if fields:
                status, reason = 'fail', 'settings_mismatch'
            strata.append({**base, **dict(zip(STRATA[6:], key)),
                **observation_stats(rows),
                'verdict': status, 'reason': reason,
                **({'settings_mismatches': fields} if fields else {}),
                'coverage': coverage(protein, cell[5], expected.get(key, set()))})
    counts = Counter(row['verdict'] for row in strata)
    return {'contract_version': contract['contract_version'], 'ticket': str(ticket),
            'thresholds': thresholds, 'strata': strata, 'verdict_counts': dict(counts),
            'report_failures': report_failures,
            'passed': bool(strata) and not report_failures and counts['pass'] == len(strata)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs='*')
    parser.add_argument('--contract', type=Path,
                        default=Path('docs/validation/43-contract/frozen-contract-v1.1.json'))
    parser.add_argument('--manifest', type=Path, nargs='*', default=[],
                        help='driver manifests retain failed and unexecuted cases')
    parser.add_argument('--ticket', choices=('3', '4'), default='3')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    paths = list(args.reports)
    failures = []
    for path in args.manifest:
        manifest = json.loads(path.read_text())
        for case in manifest['cases']:
            output = Path(case['output'])
            if case['status'] == 'completed' and output.is_file():
                payload = output.read_bytes()
                if hashlib.sha256(payload).hexdigest() != case['output_sha256']:
                    raise ValueError(f'report hash mismatch: {output}')
                paths.append(output)
            else:
                failures.append({key: case.get(key) for key in (
                    'id', 'status', 'failure_class', 'failure_phase', 'returncode', 'log', 'output')})
    paths = list(dict.fromkeys(paths))
    result = evaluate_reports([json.loads(path.read_text()) for path in paths],
                              json.loads(args.contract.read_text()), args.ticket)
    result['sources'] = [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                         for path in [args.contract, *args.manifest, *paths]]
    result['execution_failures'] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    main()
