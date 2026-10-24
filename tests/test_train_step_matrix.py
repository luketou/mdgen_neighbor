"""#43 分片矩陣、缺口與失敗分類的 CPU 反例。"""
import json
import subprocess

import pytest

from mdgen.train_step_matrix import aggregate, failure_category, make_manifest, run_shard, write_json


def test_manifest_complete_stable_and_no_candidate_refill():
    a, b = make_manifest(), make_manifest()
    assert a == b
    assert len(a['cases']) == 6642
    assert len(a['shards']) == 1620
    assert len({c['case_id'] for c in a['cases']}) == 6642
    assert sum(c['plan_status'] == 'phase1_infeasible' for c in a['cases']) == 486
    assert sum(c['plan_status'] == 'duplicate' for c in a['cases']) == 4536
    long = [c for c in a['cases'] if c['protein'] == '6sup_A' and c['halo'] == 2]
    assert {c['group'] for c in long} == {95}
    assert {c['frames'] for c in long} == {4, 64, 250}
    assert {c['t'] for c in long} == {.25, .5, .75}
    assert {c['noise_seed'] for c in long} == {0, 1, 2}
    batched = make_manifest(7)
    assert [c['case_id'] for c in batched['cases']] == [c['case_id'] for c in a['cases']]
    assert sum(len(s['case_ids']) for s in batched['shards']) == 1620
    assert len(batched['shards']) == 232


def test_aggregator_marks_missing_and_failed_shards_incomplete():
    manifest = make_manifest()
    shard = manifest['shards'][0]
    failed = dict(manifest_id=manifest['manifest_id'], shard_id=shard['shard_id'],
                  run_id='run', cases=[dict(case_id=shard['case_ids'][0], status='failed',
                                           failure_category='timeout')])
    report = aggregate(manifest, [failed])
    assert report['status'] == '未完成'
    assert len(report['missing_shards']) == 1619
    assert any('timeout' in s['failure_categories'] for s in report['strata'])
    assert any('missing_shard' in s['failure_categories'] for s in report['strata'])
    assert not any(s.get('passed') for s in report['strata'])
    with pytest.raises(ValueError, match='multiple runs'):
        aggregate(manifest, [failed, failed])
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('error,expected', [
    (RuntimeError('CUDA out of memory'), 'oom'),
    (RuntimeError('missing_gpu: CUDA unavailable'), 'missing_gpu'),
    (subprocess.TimeoutExpired('probe', 1), 'timeout'),
    (RuntimeError('shape mismatch'), 'program_error')])
def test_failure_categories_stay_separate(error, expected):
    assert failure_category(error) == expected


def test_rerun_creates_distinct_results_and_preserves_failed_case(tmp_path):
    manifest = make_manifest()
    shard_id = manifest['shards'][0]['shard_id']
    # 缺 checkpoint 仍保存該分片的 program_error，不進模型或 GPU。
    a = run_shard(manifest, shard_id, tmp_path, tmp_path, tmp_path / 'absent')
    b = run_shard(manifest, shard_id, tmp_path, tmp_path, tmp_path / 'absent')
    assert a != b and a.exists() and b.exists()
    assert json.loads(a.read_text())['cases'][0]['failure_category'] == 'program_error'
    with pytest.raises(FileExistsError):
        write_json(a, {})


def test_failed_preflight_classified_without_running_model(monkeypatch, tmp_path):
    manifest = make_manifest()
    preflight = tmp_path / 'preflight.json'
    preflight.write_text(json.dumps({'passed': False, 'error': 'Found no NVIDIA driver on your system'}))
    def forbidden(*args, **kwargs):
        pytest.fail('failed preflight must not run model subprocess')
    monkeypatch.setattr(subprocess, 'run', forbidden)
    path = run_shard(manifest, manifest['shards'][0]['shard_id'], tmp_path,
                     tmp_path, tmp_path / 'absent', preflight_report=preflight)
    result = json.loads(path.read_text())
    assert result['status'] == 'failed'
    assert result['cases'][0]['stage'] == 'preflight'
    assert result['cases'][0]['failure_category'] == 'missing_gpu'


def test_summary_keeps_repeat_count_separate_from_spatial_samples():
    from mdgen.train_step_matrix import summarize
    comparison = {'loss': dict(signed_difference=1., absolute_difference=1., relative_difference=None),
                  'gradients': {}, 'updates': {}}
    measurement = {'warmup': False, 'comparison': comparison,
                   'full': {'step_time_s': 2.}, 'candidate': {'step_time_s': 1.}}
    record = {'case_id': 'case', 'status': 'descriptive', 'report': {
        'alignment': {'loss_group_ids': [1], 'loss_residue_ids': [2, 3], 'full_length': 10},
        'measurements': [measurement, measurement, measurement]}}
    stats = summarize([record])
    assert stats['metrics']['loss.absolute_difference']['effective_samples'] == 1
    assert stats['metrics']['loss.absolute_difference']['measurement_count'] == 3
    assert stats['metrics']['full.step_time_s']['median'] == 2
    assert stats['loss_residue_union_fraction'] == .2


def test_manifest_changes_are_rejected_before_execution_or_aggregation(tmp_path):
    from mdgen.train_step_matrix import validate_manifest
    manifest = make_manifest()
    manifest['cases'][0]['t'] = .9
    with pytest.raises(ValueError, match='frozen matrix'):
        validate_manifest(manifest)
    with pytest.raises(ValueError, match='frozen matrix'):
        aggregate(manifest, [])
    with pytest.raises(ValueError, match='frozen matrix'):
        run_shard(manifest, manifest['shards'][0]['shard_id'], tmp_path, tmp_path, tmp_path / 'absent')
    assert not list(tmp_path.iterdir())


def summary_record(identity='case', status='passed'):
    metrics = dict(difference_rms=3., reference_zero=True, candidate_zero=False,
                   one_sided_zero=True, near_zero_reference=True, nonfinite=0)
    comparison = {'loss': dict(signed_difference=1., absolute_difference=1., relative_difference=None),
                  'gradients': {'group': metrics}, 'updates': {'group': metrics.copy()}}
    measurement = {'warmup': False, 'comparison': comparison,
                   'full': {'step_time_s': 2., 'loss_finite': True},
                   'candidate': {'step_time_s': 1., 'loss_finite': True}}
    return {'case_id': identity, 'status': status, 'report': {
        'alignment': {'loss_group_ids': [1], 'loss_residue_ids': [2, 3], 'full_length': 10},
        'measurements': [measurement]}}


@pytest.mark.parametrize('category', ['control_failure', 'nonfinite'])
def test_dedup_preserves_failed_status_and_evidence(category):
    manifest = make_manifest()
    cases = [c for c in manifest['cases'] if c['plan_status'] == 'ready' and c['protein'] == '6sup_A'
             and c['frames'] == 64 and c['halo'] == 1 and c['t'] == .25
             and c['noise_seed'] == 0 and c['control_type'] == 'paired'][:2]
    results = []
    for index, case in enumerate(cases):
        shard = next(s for s in manifest['shards'] if case['case_id'] in s['case_ids'])
        record = summary_record(case['case_id'], 'failed' if index else 'descriptive')
        record['failure_category'] = category
        results.append(dict(manifest_id=manifest['manifest_id'], shard_id=shard['shard_id'], cases=[record]))
    report = aggregate(manifest, results)
    group = next(g for g in report['strata'] if any(c['case_id'] == cases[1]['case_id'] for c in g['cases']))
    duplicate = next(c for c in group['cases'] if c['case_id'] == cases[1]['case_id'])
    assert duplicate['status'] == 'failed'
    assert duplicate['failure_category'] == category
    assert duplicate['deduplicated'] is True
    assert duplicate['duplicate_of'] == cases[0]['case_id']
    assert group['status'] == '未完成'
    assert group['statistics']['metrics']['loss.absolute_difference']['measurement_count'] == 1
    assert not group['statistics']['failed_diagnostics']['metrics']


def test_failed_diagnostics_counts_flags_and_filters_nonfinite_quantiles():
    from mdgen.train_step_matrix import summarize
    failed = summary_record('failed', 'failed')
    bad = summary_record('nonfinite', 'failed')
    bad['report']['measurements'][0]['comparison']['loss']['absolute_difference'] = float('nan')
    bad['report']['measurements'][0]['comparison']['gradients']['group'].update(
        difference_rms=float('inf'), reference_zero=None, nonfinite=2)
    stats = summarize([summary_record(), failed, bad])
    assert stats['metrics']['loss.absolute_difference']['measurement_count'] == 1
    diagnostic = stats['failed_diagnostics']
    assert diagnostic['metrics']['loss.absolute_difference']['median'] == 1
    assert diagnostic['metrics']['loss.absolute_difference']['measurement_count'] == 1
    assert diagnostic['counts']['loss.absolute_difference.nonfinite'] == 1
    assert diagnostic['counts']['gradients.group.nonfinite'] == 2
    assert diagnostic['counts']['gradients.group.reference_zero'] == 1
    assert diagnostic['counts']['updates.group.one_sided_zero'] == 2
    assert diagnostic['counts']['updates.group.near_zero_reference'] == 2
    assert 'gradients.group.reference_zero' not in diagnostic['metrics']
    json.dumps(stats, allow_nan=False)


@pytest.mark.parametrize('message', [
    'CUDA error: no CUDA-capable device is detected',
    'cudaErrorNoDevice: no CUDA-capable device is detected',
    'CUDA error: no devices supporting CUDA',
    'CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected'])
def test_real_cuda_no_device_errors(message):
    assert failure_category(RuntimeError(message)) == 'missing_gpu'


def test_preflight_structured_category_takes_precedence(tmp_path):
    manifest = make_manifest()
    preflight = tmp_path / 'structured.json'
    preflight.write_text(json.dumps({'passed': False, 'failure_category': 'missing_gpu',
                                    'error': 'localized driver message'}))
    path = run_shard(manifest, manifest['shards'][0]['shard_id'], tmp_path,
                     tmp_path, tmp_path / 'absent', preflight_report=preflight)
    assert json.loads(path.read_text())['cases'][0]['failure_category'] == 'missing_gpu'


@pytest.mark.parametrize('mode', ['all', 'none'])
def test_matrix_raw_tensor_option_reaches_probe_command(tmp_path, monkeypatch, mode):
    from mdgen.train_step_matrix import case_command, main, CONTRACT_DIR
    manifest = make_manifest()
    case = next(c for c in manifest['cases'] if c['plan_status'] == 'ready')
    inventory = json.loads((CONTRACT_DIR / 'infeasible-groups.json').read_text())['proteins']
    length = inventory[f"{case['protein']}_R{case['replica']}"]['residues']
    (tmp_path / 'validation.csv').write_text(f"name,seqres\n{case['protein']},{'A' * length}\n")
    command = case_command(case, tmp_path, 'checkpoint', tmp_path / 'probe.json', mode)
    assert command[command.index('--raw-tensors') + 1] == mode
    shard_result = tmp_path / 'shard.json'
    shard_result.write_text('{"status":"completed"}')
    def run(*args):
        assert args[-1] == mode
        return shard_result
    monkeypatch.setattr('mdgen.train_step_matrix.run_shard', run)
    main(['run', '--shard-id', manifest['shards'][0]['shard_id'], '--output-root', str(tmp_path),
          '--data-root', str(tmp_path), '--checkpoint', 'checkpoint', '--raw-tensors', mode])


def test_matrix_run_omitting_raw_tensors_saves_none(tmp_path, monkeypatch):
    from mdgen.train_step_matrix import case_command, main
    manifest = make_manifest()
    case = next(c for c in manifest['cases'] if c['plan_status'] == 'ready')
    assert case_command.__defaults__[-1] == 'none'
    shard_result = tmp_path / 'shard.json'
    shard_result.write_text('{"status":"completed"}')
    def run(*args):
        assert args[-1] == 'none'
        return shard_result
    monkeypatch.setattr('mdgen.train_step_matrix.run_shard', run)
    main(['run', '--shard-id', manifest['shards'][0]['shard_id'], '--output-root', str(tmp_path),
          '--data-root', str(tmp_path), '--checkpoint', 'checkpoint'])
