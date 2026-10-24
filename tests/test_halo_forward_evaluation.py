"""Frozen forward policy counterexamples on raw Halo reports."""
import copy
import json
from pathlib import Path

import pytest

from mdgen.full_length.forward_evaluation import evaluate_reports

CONTRACT = json.loads((Path(__file__).resolve().parents[1] /
                       'docs/validation/43-contract/frozen-contract-v1.1.json').read_text())


def inputs(truncated=False, category='regular', absolute=0.0, relative=0.0):
    contract = copy.deepcopy(CONTRACT)
    contract['forward_matrix'].update({'tickets': {'3': ['prot']}, 'replicas': [1],
        'frames': {'4': 'diagnostic'}, 't': [0.5], 'noise_seeds': [0]})
    contract['seed_groups']['halo_rings'] = [1]
    entry = {'region_id': 0, 'loss_residue_indices': [0], 'boundary_rings': [0],
             'paired_common_subset': True, 'truncated': truncated}
    row = {'region_id': 0, 'residue_index': 0, 't': 0.5, 'noise_seed': 0, 'halo_rings': 1,
        'boundary_ring': 0, 'truncated': truncated, 'paired_common_subset': True,
        'reference_class': category, 'absolute_rms': absolute, 'relative': relative,
        'reference_norm': 1.0, 'l2': absolute, 'effective_components': 1}
    protein = {'name': 'prot_R1', 'residues': 2, 'regions': [{'by_halo': {'1': entry}}],
        'observations': [row], 'candidates': [{'by_halo': {'1': {
            'status': 'selected', 'region_id': 0, 'paired_common_subset': True}}}],
        'nonfinite_counts': {'input': 0, 'full_output': 0, 'region_output': 0}}
    fixed = contract['fixed_settings']
    model = {('spatial_' + key if key in ('cell_width', 'contact_radius') else key): value
             for key, value in fixed['local_spatial'].items()}
    model.update({'num_frames': 4, 'dropout': fixed['dropout']})
    return [{'checkpoint_sha256': fixed['checkpoint_sha256'],
        'precision_state': {'precision_config': fixed['precision'],
            'float32_matmul_precision': fixed['float32_matmul_precision'],
            'cuda_matmul_allow_tf32': fixed['tf32'], 'cudnn_allow_tf32': fixed['tf32']},
        'settings': {'replicate': 1, 'num_frames': 4, 'times': [0.5], 'model': model,
            'noise_seeds': [0], 'halo_rings': [1], 'selection_seed': 43002, 'regions': 8,
            'region_residues': 896, 'warmup': 1, 'repeats': 3},
        'proteins': [protein]}], contract


@pytest.mark.parametrize('truncated,absolute,relative,status', [
    (False, 0, 0, 'pass'), (False, 2e-5, 0, 'fail'), (False, 0, 2e-5, 'fail'),
    (True, 0.02, 0, 'fail'), (True, 0, 0.02, 'fail'), (True, 0.001, 0.001, 'pass')])
def test_thresholds_keep_full_coverage_separate(truncated, absolute, relative, status):
    reports, contract = inputs(truncated, absolute=absolute, relative=relative)
    row, = evaluate_reports(reports, contract)['strata']
    assert row['verdict'] == status
    assert row['full_coverage_or_truncated'] == ('truncated' if truncated else 'full_coverage')
    assert row['coverage']['loss_residue_union_fraction'] == 0.5
    if status == 'fail':
        assert row['reason'] == 'threshold'


@pytest.mark.parametrize('category', ['zero_reference', 'near_zero_reference'])
def test_undefined_relative_still_requires_absolute(category):
    reports, contract = inputs(category=category, relative=None)
    row, = evaluate_reports(reports, contract)['strata']
    assert row['verdict'] == 'not_evaluable' and row[category] == 1
    reports[0]['proteins'][0]['observations'][0]['absolute_rms'] = 1
    assert evaluate_reports(reports, contract)['strata'][0]['verdict'] == 'fail'


def test_nonfinite_never_disappears_from_verdict():
    reports, contract = inputs(category='nonfinite', absolute=None, relative=None)
    row, = evaluate_reports(reports, contract)['strata']
    assert row['verdict'] == 'fail' and row['reason'] == 'nonfinite'


@pytest.mark.parametrize('missing', ['report', 'protein', 'region', 'observation', 'time', 'seed'])
def test_missing_required_results_are_incomplete(missing):
    reports, contract = inputs()
    if missing == 'report':
        reports = []
    elif missing == 'protein':
        reports[0]['proteins'] = []
    elif missing == 'region':
        reports[0]['proteins'][0]['regions'] = []
    elif missing == 'observation':
        reports[0]['proteins'][0]['observations'] = []
    elif missing == 'time':
        contract['forward_matrix']['t'].append(0.75)
    else:
        contract['forward_matrix']['noise_seeds'].append(1)
    rows = evaluate_reports(reports, contract)['strata']
    assert any(row['verdict'] == 'incomplete' for row in rows)


def test_boundaries_are_never_pooled_to_hide_an_outlier():
    reports, contract = inputs(truncated=True)
    protein = reports[0]['proteins'][0]
    entry = protein['regions'][0]['by_halo']['1']
    entry['loss_residue_indices'].append(1)
    entry['boundary_rings'].append(1)
    protein['observations'].append({**protein['observations'][0], 'residue_index': 1,
                                    'boundary_ring': 1, 'absolute_rms': 0.02})
    rows = evaluate_reports(reports, contract)['strata']
    assert [row['verdict'] for row in rows] == ['pass', 'fail']


def test_missing_a_whole_region_cannot_pass_the_remaining_region():
    reports, contract = inputs()
    protein = reports[0]['proteins'][0]
    protein['candidates'].append({'by_halo': {'1': {'status': 'selected', 'region_id': 1}}})
    row, = evaluate_reports(reports, contract)['strata']
    assert row['verdict'] == 'incomplete'


def test_null_required_metric_is_incomplete():
    reports, contract = inputs(absolute=None)
    assert evaluate_reports(reports, contract)['strata'][0]['verdict'] == 'incomplete'


def test_evaluator_cli_keeps_failure_classes_and_source_hashes(tmp_path):
    import hashlib
    from mdgen.full_length.forward_evaluation import main
    reports, contract = inputs()
    raw = tmp_path / 'raw.json'
    raw.write_text(json.dumps(reports[0]))
    frozen = tmp_path / 'contract.json'
    frozen.write_text(json.dumps(contract))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'cases': [
        {'id': 'completed', 'status': 'completed', 'output': str(raw),
         'output_sha256': hashlib.sha256(raw.read_bytes()).hexdigest()},
        {'id': 'missing', 'status': 'failed', 'output': str(tmp_path / 'missing.json'),
         'failure_class': 'oom', 'returncode': 1, 'failure_phase': 'forward_or_setup',
         'log': 'case.log'}]}))
    output = tmp_path / 'evaluation.json'
    result = main(['--contract', str(frozen), '--manifest', str(manifest), '--output', str(output)])
    assert result['execution_failures'][0]['failure_class'] == 'oom'
    assert len(result['sources']) == 3
    assert json.loads(output.read_text()) == result
    raw.write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        main(['--contract', str(frozen), '--manifest', str(manifest), '--output', str(output)])


def test_nonfinite_measurement_is_scoped_to_its_noise_seed():
    reports, contract = inputs()
    contract['forward_matrix']['noise_seeds'] = [0, 1]
    report = reports[0]
    report['settings']['noise_seeds'] = [0, 1]
    protein = report['proteins'][0]
    protein['observations'].append({**protein['observations'][0], 'noise_seed': 1})
    protein['nonfinite_counts']['full_output'] = 1
    protein['measurements'] = {'full': [
        {'t': 0.5, 'noise_seed': 0, 'nonfinite': 1},
        {'t': 0.5, 'noise_seed': 1, 'nonfinite': 0}], 'regions': []}
    assert [row['verdict'] for row in evaluate_reports(reports, contract)['strata']] == [
        'fail', 'pass']


def test_max_threshold_cannot_be_hidden_by_a_passing_p95():
    reports, contract = inputs(truncated=True)
    protein = reports[0]['proteins'][0]
    entry = protein['regions'][0]['by_halo']['1']
    entry.update({'loss_residue_indices': list(range(100)), 'boundary_rings': [0] * 100})
    prototype = protein['observations'][0]
    protein['observations'] = [{**prototype, 'residue_index': i,
        'absolute_rms': 0.06 if i == 99 else 0.0} for i in range(100)]
    protein['residues'] = 200
    row, = evaluate_reports(reports, contract)['strata']
    assert row['absolute_rms']['p95'] == 0 and row['verdict'] == 'fail'


def test_missing_frozen_candidate_is_incomplete_even_when_a_region_remains():
    reports, contract = inputs()
    reports[0]['proteins'][0]['groups'] = 10
    assert evaluate_reports(reports, contract)['strata'][0]['verdict'] == 'incomplete'


@pytest.mark.parametrize('field,value', [
    ('checkpoint_sha256', 'wrong'), ('settings.selection_seed', 7),
    ('settings.regions', 1), ('settings.region_residues', 80),
    ('settings.warmup', 0), ('settings.repeats', 1), ('settings.halo_rings', [1, 2]),
    ('precision_state.precision_config', 'bf16-train'),
    ('precision_state.float32_matmul_precision', 'high'),
    ('precision_state.cuda_matmul_allow_tf32', True),
    ('precision_state.cudnn_allow_tf32', True), ('settings.model.dropout', 0.1),
    ('settings.model.spatial_cell_width', 8.0),
    ('settings.model.spatial_contact_radius', 8.0), ('settings.model.core_capacity', 8),
    ('settings.model.ghost_capacity', 8), ('settings.model.cell_summaries', True),
    ('settings.model.group_batch_size', 1), ('settings.model.ffn_chunk_size', 8192),
    ('settings.model.num_frames', 250), ('settings.num_frames', 250)])
def test_report_with_wrong_fixed_settings_fails(field, value):
    reports, contract = inputs()
    target = reports[0]
    parts = field.split('.')
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    result = evaluate_reports(reports, contract)
    assert not result['passed']
    failure, = result['report_failures']
    assert failure['reason'] == 'settings_mismatch' and field in failure['fields']
    if field != 'settings.num_frames':
        assert all(row['verdict'] == 'fail' and row['reason'] == 'settings_mismatch'
                   for row in result['strata'])


def test_missing_fixed_settings_are_not_accepted():
    reports, contract = inputs()
    del reports[0]['settings']['warmup']
    del reports[0]['precision_state']
    result = evaluate_reports(reports, contract)
    assert result['strata'][0]['reason'] == 'settings_mismatch'
    assert 'settings.warmup' in result['report_failures'][0]['fields']
    assert 'precision_state.precision_config' in result['report_failures'][0]['fields']


def test_all_infeasible_is_distinct_from_a_missing_result():
    reports, contract = inputs()
    protein = reports[0]['proteins'][0]
    protein.update({'regions': [], 'observations': [], 'groups': 10,
        'candidates': [{'by_halo': {'1': {'status': 'infeasible', 'reason': 'too big'}}}
                       for _ in range(8)]})
    result = evaluate_reports(reports, contract)
    row, = result['strata']
    assert row['verdict'] == 'no_feasible_region' and not result['passed']
    assert row['coverage']['attempted'] == row['coverage']['skipped'] == 8
    assert row['coverage']['feasible'] == row['coverage']['deduplicated_regions'] == 0
    assert evaluate_reports([], contract)['strata'][0]['verdict'] == 'incomplete'
    protein['candidates'].pop()
    assert evaluate_reports(reports, contract)['strata'][0]['verdict'] == 'incomplete'


def test_not_evaluable_stratum_blocks_overall_pass():
    reports, contract = inputs()
    protein = reports[0]['proteins'][0]
    entry = protein['regions'][0]['by_halo']['1']
    entry['loss_residue_indices'].append(1)
    entry['boundary_rings'].append(1)
    protein['observations'].append({**protein['observations'][0], 'residue_index': 1,
        'boundary_ring': 1, 'reference_class': 'zero_reference', 'relative': None})
    result = evaluate_reports(reports, contract)
    assert [row['verdict'] for row in result['strata']] == ['pass', 'not_evaluable']
    assert result['passed'] is False


def long_inputs(deep_feasible_groups):
    """Ticket 4 shape: two candidates, halo 2 feasible only for those listed."""
    reports, contract = inputs(truncated=True, absolute=0.001, relative=0.001)
    contract['forward_matrix']['tickets'] = {'4': ['prot']}
    contract['seed_groups']['halo_rings'] = [1, 2]
    report = reports[0]
    report['settings']['halo_rings'] = [1, 2]
    protein = report['proteins'][0]
    protein.update({'residues': 10, 'regions': [], 'observations': [], 'candidates': []})
    for rid, group in enumerate((95, 7)):
        deep = group in deep_feasible_groups
        by_halo = {}
        for halo in ((1, 2) if deep else (1,)):
            by_halo[str(halo)] = {'region_id': rid, 'loss_residue_indices': [rid],
                'boundary_rings': [0], 'paired_common_subset': deep, 'truncated': True,
                'region_residues': 3 + halo}
            protein['observations'].append({**inputs(True)[0][0]['proteins'][0]['observations'][0],
                'region_id': rid, 'residue_index': rid, 'halo_rings': halo,
                'paired_common_subset': deep, 'absolute_rms': 0.001 * (1 + 3 * (not deep)),
                'relative': 0.001})
        protein['regions'].append({'by_halo': by_halo})
        protein['candidates'].append({'seed_group': group, 'by_halo': {
            '1': {'status': 'selected', 'region_id': rid, 'paired_common_subset': deep},
            '2': ({'status': 'selected', 'region_id': rid, 'paired_common_subset': True}
                  if deep else {'status': 'infeasible', 'reason': 'too big'})}})
    return reports, contract


def test_long_protein_reports_context_and_common_subset_bias():
    reports, contract = long_inputs([95])
    contract['seed_groups']['long_protein_halo2_feasible_candidates'] = [95]
    result = evaluate_reports(reports, contract, ticket='4')
    assert result['passed'] and not result['report_failures']
    halo1 = [row for row in result['strata'] if row['halo_rings'] == 1]
    assert {r['region_id']: r['outside_context_residues']
            for row in halo1 for r in row['coverage']['regions']} == {0: 6, 1: 6}
    assert {row['paired_common_subset'] for row in halo1} == {True, False}
    halo2, = [row for row in result['strata'] if row['halo_rings'] == 2]
    assert halo2['coverage']['skipped'] == 1 and halo2['coverage']['feasible'] == 1
    bias, = result['common_subset_bias']
    assert bias['all']['samples'] == 2 and bias['paired_common_subset']['samples'] == 1
    # The common subset is the easier region here; the bias must show it, not average it away.
    assert bias['common_minus_all']['absolute_rms']['max'] == pytest.approx(-0.003)


@pytest.mark.parametrize('feasible', [[], [95, 7]])
def test_long_protein_deep_halo_candidates_must_match_the_contract(feasible):
    reports, contract = long_inputs(feasible)
    contract['seed_groups']['long_protein_halo2_feasible_candidates'] = [95]
    result = evaluate_reports(reports, contract, ticket='4')
    failure, = result['report_failures']
    assert failure['reason'] == 'deep_halo_candidates_mismatch'
    assert sorted(failure['actual']) == sorted(feasible) and not result['passed']


def test_missing_halo_settings_fails_the_report_without_crashing():
    reports, contract = inputs()
    del reports[0]['settings']['halo_rings']
    result = evaluate_reports(reports, contract)
    failure, = result['report_failures']
    assert failure['verdict'] == 'fail'
    assert failure['reason'] == 'settings_mismatch'
    assert 'settings.halo_rings' in failure['fields']
    assert result['passed'] is False
