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
