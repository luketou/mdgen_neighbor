"""Case planning and stop rules of the Issue #43 pilot driver; no model runs."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('issue43_pilot', ROOT / 'scripts/issue43_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_memory_spec_defaults_match_the_approved_local_settings():
    assert pilot.parse_memory('') == {'group_batch_size': 4, 'ffn_chunk_size': 65536,
                                      'expandable_segments': False}


def test_memory_spec_overrides_each_setting():
    assert pilot.parse_memory('gbs=1,ffn=8192,expandable') == {
        'group_batch_size': 1, 'ffn_chunk_size': 8192, 'expandable_segments': True}


@pytest.mark.parametrize('text', ['gbs=0', 'ffn=-1', 'gbs=x', 'chunk=4', 'expandable=1'])
def test_memory_spec_rejects_unknown_or_invalid_settings(text):
    with pytest.raises(ValueError):
        pilot.parse_memory(text)


def test_step_spec_carries_frames_and_memory():
    assert pilot.parse_step('250:gbs=1') == {'frames': 250, 'group_batch_size': 1,
                                             'ffn_chunk_size': 65536,
                                             'expandable_segments': False}
    assert pilot.parse_step('64')['frames'] == 64


@pytest.mark.parametrize('text', ['0', '-4', 'x', '64:'])
def test_step_spec_rejects_invalid_frames(text):
    with pytest.raises(ValueError):
        pilot.parse_step(text)


def test_local_args_use_the_requested_memory_settings():
    args = pilot.local_args(pilot.parse_memory('gbs=1,ffn=8192'))
    assert args[args.index('--group_batch_size') + 1] == '1'
    assert args[args.index('--ffn_chunk_size') + 1] == '8192'
    assert pilot.local_args(pilot.parse_memory('')) == pilot.LOCAL


def test_child_env_sets_allocator_only_for_expandable_segments():
    assert 'PYTORCH_CUDA_ALLOC_CONF' not in pilot.child_env(pilot.parse_memory(''), {})
    env = pilot.child_env(pilot.parse_memory('expandable'), {'PATH': '/bin'})
    assert env == {'PATH': '/bin', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}


def _done(step, status, failure_class=None):
    case = {'step': pilot.parse_step(step), 'status': status}
    if failure_class:
        case['failure_class'] = failure_class
    return case


def test_more_frames_with_the_same_memory_settings_are_skipped_after_oom():
    finished = [_done('64', 'completed'), _done('96', 'failed', 'oom')]
    assert pilot.skip_reason(pilot.parse_step('128'), finished) == 'more_frames_than_oom_step'
    assert pilot.skip_reason(pilot.parse_step('96:gbs=1'), finished) is None


def test_non_oom_failure_does_not_skip_larger_steps():
    finished = [_done('96', 'failed', 'program_error')]
    assert pilot.skip_reason(pilot.parse_step('128'), finished) is None


def test_further_memory_settings_are_skipped_once_the_frames_fit():
    finished = [_done('250:gbs=1', 'completed')]
    assert pilot.skip_reason(pilot.parse_step('250:gbs=1,ffn=8192'), finished) == \
        'frames_already_fit'
    assert pilot.skip_reason(pilot.parse_step('128'), finished) is None


def test_select_cases_keeps_planned_order_and_rejects_unknown_ids():
    cases = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
    assert [c['id'] for c in pilot.select_cases(cases, ['c', 'a'])] == ['a', 'c']
    assert pilot.select_cases(cases, None) == cases
    with pytest.raises(ValueError, match='unknown case'):
        pilot.select_cases(cases, ['a', 'z'])


def _pilot_inputs(tmp_path, monkeypatch):
    import hashlib
    root = tmp_path / 'data'
    (root / 'arrays').mkdir(parents=True)
    (root / 'partitions').mkdir()
    (root / 'validation.csv').write_text('name,seqres\n' + ''.join(
        f'{protein},{"G" * length}\n' for protein, length in zip(pilot.PROTEINS, pilot.LENGTHS)))
    for replica in (1, 2, 3):
        for protein in pilot.PROTEINS:
            stem = f'{protein}_R{replica}_i400ps'
            for path in (root / 'arrays' / f'{stem}.npy', root / 'arrays' / f'{stem}_chain.npz',
                         root / 'partitions' / f'{stem}_partition.npz'):
                path.write_bytes(b'')
    checkpoint = tmp_path / 'atlas.ckpt'
    checkpoint.write_bytes(b'weights')
    monkeypatch.setattr(pilot, 'CHECKPOINT_SHA256', hashlib.sha256(b'weights').hexdigest())
    monkeypatch.delenv('MDGEN_RUN_DIR', raising=False)
    return root, checkpoint


def test_sweep_driver_applies_stop_rules_and_records_memory(tmp_path, monkeypatch):
    import json
    import subprocess
    root, checkpoint = _pilot_inputs(tmp_path, monkeypatch)
    calls = []

    def fake_run(command, stdout, stderr, timeout, env):
        frames = int(command[command.index('--num_frames') + 1])
        gbs = command[command.index('--group_batch_size') + 1]
        calls.append((frames, gbs, env.get('PYTORCH_CUDA_ALLOC_CONF')))
        if frames >= 96 and gbs == '4':
            stdout.write('loss.backward()\ntorch.OutOfMemoryError: CUDA out of memory\n')
            return subprocess.CompletedProcess(command, 1)
        output = Path(command[command.index('--output') + 1])
        output.write_text(json.dumps({'runs': []}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pilot.subprocess, 'run', fake_run)
    report = pilot.main(['--mode', 'sweep', '--protein', '6sup_A', '--replica', '1',
                         '--data_root', str(root), '--checkpoint', str(checkpoint),
                         '--output_dir', str(tmp_path / 'out'), '--budget_minutes', '57',
                         '--step', '64', '--step', '96', '--step', '128',
                         '--step', '250:gbs=1', '--step', '250:gbs=1,ffn=8192',
                         '--step', '250:gbs=1,ffn=8192,expandable'])
    status = {case['id']: (case['status'], case.get('reason'), case.get('failure_class'))
              for case in report['cases']}
    assert status == {
        '6sup_A-R1-full-f64': ('completed', None, None),
        '6sup_A-R1-full-f96': ('failed', None, 'oom'),
        '6sup_A-R1-full-f128': ('skipped_by_rule', 'more_frames_than_oom_step', None),
        '6sup_A-R1-full-f250-gbs1-ffn65536': ('completed', None, None),
        '6sup_A-R1-full-f250-gbs1-ffn8192': ('skipped_by_rule', 'frames_already_fit', None),
        '6sup_A-R1-full-f250-gbs1-ffn8192-expandable':
            ('skipped_by_rule', 'frames_already_fit', None)}
    assert calls == [(64, '4', None), (96, '4', None), (250, '1', None)]
    saved = json.loads((tmp_path / 'out' / 'manifest.json').read_text())
    assert saved['driver_finished'] and saved['mode'] == 'sweep'
    assert saved['cases'][3]['memory'] == {'group_batch_size': 1, 'ffn_chunk_size': 65536,
                                           'expandable_segments': False}
    assert saved['cases'][1]['failure_phase'] == 'backward'


def _one_candidate(*args):
    return [{'setting': 'halo1', 'seed_group': 7, 'loss_group_ids': [0],
             'region_residues': 10, 'truncated': False}]


def test_memory_steps_still_run_after_a_default_memory_oom(tmp_path, monkeypatch):
    import subprocess
    root, checkpoint = _pilot_inputs(tmp_path, monkeypatch)

    def fake_run(command, stdout, stderr, timeout, env):
        stdout.write('torch.OutOfMemoryError: CUDA out of memory\n')
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(pilot.subprocess, 'run', fake_run)
    report = pilot.main(['--mode', 'sweep', '--protein', '6sup_A', '--replica', '1',
                         '--data_root', str(root), '--checkpoint', str(checkpoint),
                         '--output_dir', str(tmp_path / 'out'),
                         '--step', '64', '--step', '96', '--step', '250:gbs=1'])
    assert [case['status'] for case in report['cases']] == ['failed', 'skipped_by_rule', 'failed']
    assert report['memory_settings'] is None
    assert all(case['timeout_s'] == pilot.SWEEP_CASE_TIMEOUT_S for case in report['cases'])


VALIDATION = ROOT / 'docs/validation/43-49501/results/training_region_probe_49501/manifest.json'


@pytest.mark.parametrize('case_id', ['6sup_A-R1-full', '6sms_A-R1-coverage_control-group55',
                                     '6sup_A-R1-halo2-group95'])
def test_default_train_commands_match_the_recorded_49501_commands(case_id, tmp_path,
                                                                   monkeypatch):
    import json
    recorded = next(case for case in json.loads(VALIDATION.read_text())['cases']
                    if case['id'] == case_id)
    expected = list(recorded['command'])
    where = expected.index('--output')
    del expected[where:where + 2]
    server_root = '/share/home/r15543056/MDgen/mdgen/data/atlas_validation_ticket1_20261023b'
    root = tmp_path / 'data'
    root.mkdir()
    (root / 'validation.csv').write_text('name,seqres\n6sms_A,A\n6sup_A,G\n')
    monkeypatch.chdir(tmp_path)
    output_dir = Path('results/training_region_probe_49501')
    output_dir.mkdir(parents=True)
    command = pilot.train_command(
        root, Path('/share/home/r15543056/MDgen/mdgen/ckpt/atlas.ckpt'), output_dir, case_id,
        recorded['protein'], recorded['replica'], 250, recorded['setting'],
        recorded.get('seed_group'), pilot.DEFAULT_MEMORY)
    command = [expected[0], *(arg.replace(str(root), server_root) for arg in command[1:])]
    assert command == expected


def test_train_driver_rejects_unknown_selected_cases(tmp_path, monkeypatch):
    root, checkpoint = _pilot_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(pilot, 'training_candidates', _one_candidate)
    with pytest.raises(ValueError, match='unknown case'):
        pilot.main(['--mode', 'train', '--frames', '96', '--data_root', str(root),
                    '--checkpoint', str(checkpoint), '--output_dir', str(tmp_path / 'out'),
                    '--plan_only', '--cases', '6sup_A-R1-full', 'nope'])


def test_train_driver_plans_selected_cases_at_any_frame_count(tmp_path, monkeypatch):
    root, checkpoint = _pilot_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(pilot, 'training_candidates', _one_candidate)
    report = pilot.main(['--mode', 'train', '--frames', '96', '--memory', 'gbs=1',
                         '--data_root', str(root), '--checkpoint', str(checkpoint),
                         '--output_dir', str(tmp_path / 'out'), '--plan_only',
                         '--cases', '6sup_A-R3-full', '6sup_A-R2-full'])
    assert [case['id'] for case in report['cases']] == ['6sup_A-R2-full', '6sup_A-R3-full']
    command = report['cases'][0]['command']
    assert command[command.index('--num_frames') + 1] == '96'
    assert command[command.index('--group_batch_size') + 1] == '1'
    assert all(case['status'] == 'planned_only' for case in report['cases'])


def test_driver_rejects_a_split_whose_sequence_length_differs_from_the_crop(tmp_path,
                                                                           monkeypatch):
    root, checkpoint = _pilot_inputs(tmp_path, monkeypatch)
    (root / 'validation.csv').write_text('name,seqres\n' + ''.join(
        f'{protein},{"G" * (length - (protein == "6sup_A"))}\n'
        for protein, length in zip(pilot.PROTEINS, pilot.LENGTHS)))
    with pytest.raises(ValueError, match='6sup_A has 2127 residues'):
        pilot.main(['--mode', 'train', '--frames', '4', '--data_root', str(root),
                    '--checkpoint', str(checkpoint), '--output_dir', str(tmp_path / 'out'),
                    '--plan_only'])


def test_candidate_seed_groups_keep_the_approved_permutation():
    import torch
    from mdgen.full_length.group_region import CANDIDATE_SELECTION_SEED, candidate_seed_groups
    assert pilot.SELECTION_SEED == CANDIDATE_SELECTION_SEED == 43002
    expected = torch.randperm(190, generator=torch.Generator().manual_seed(43002))[:8].tolist()
    assert candidate_seed_groups(190, 8) == expected


def test_train_mode_requires_positive_frames(tmp_path):
    with pytest.raises(SystemExit):
        pilot.main(['--mode', 'train', '--frames', '0', '--data_root', str(tmp_path),
                    '--checkpoint', str(tmp_path / 'x'), '--output_dir', str(tmp_path / 'o')])


def test_sweep_plans_full_protein_training_steps(tmp_path):
    (tmp_path / 'validation.csv').write_text('name,seqres\n6sms_A,AAA\n6sup_A,GGG\n')
    out = tmp_path / 'out'
    out.mkdir()
    cases = pilot.sweep_cases(tmp_path, Path('atlas.ckpt'), out, '6sup_A', 2,
                              [pilot.parse_step('64'), pilot.parse_step('250:gbs=1,expandable')])
    assert [c['id'] for c in cases] == ['6sup_A-R2-full-f64',
                                        '6sup_A-R2-full-f250-gbs1-ffn65536-expandable']
    first, second = cases
    command = first['command']
    assert command[command.index('--num_frames') + 1] == '64'
    assert command[command.index('--crop') + 1] == '2128'
    assert command[command.index('--replicates') + 1] == '2'
    assert '--training_region' not in command and '--grad_checkpointing' in command
    assert first['setting'] == 'full' and first['frames'] == 64
    assert second['step']['expandable_segments'] is True
    command = second['command']
    assert command[command.index('--group_batch_size') + 1] == '1'
    assert (out / '6sup_A.csv').read_text().splitlines()[1] == '6sup_A,GGG'


def test_ticket3_plan_executes_the_frozen_paired_matrix(tmp_path, monkeypatch):
    root, checkpoint = _pilot_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(pilot.subprocess, 'run', lambda *a, **k: pytest.fail('plan ran a model'))
    report = pilot.main(['--mode', 'baseline_forward', '--plan_only',
                         '--data_root', str(root), '--checkpoint', str(checkpoint),
                         '--output_dir', str(tmp_path / 'out')])
    assert len(report['cases']) == 162
    assert report['source_manifest']
    axes = set()
    for case in report['cases']:
        cmd = case['command']
        assert '--capacity_only' not in cmd
        assert cmd[cmd.index('--halo_rings') + 1:cmd.index('--halo_rings') + 3] == ['1', '2']
        assert cmd[cmd.index('--regions') + 1] == '8'
        assert cmd[cmd.index('--selection_seed') + 1] == '43002'
        assert cmd[cmd.index('--warmup') + 1] == '1'
        assert cmd[cmd.index('--repeats') + 1] == '3'
        assert '--require_gpu' in cmd
        axes.add((case['protein'], case['replica'], case['frames'], case['t'], case['noise_seed']))
    from itertools import product
    assert axes == set(product(pilot.PROTEINS[:3], (1, 2, 3), (4, 250),
                               (0.25, 0.5, 0.75), (0, 1, 2)))


def test_ticket4_plan_is_only_the_long_protein_paired_matrix(tmp_path, monkeypatch):
    root, checkpoint = _pilot_inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(pilot.subprocess, 'run', lambda *a, **k: pytest.fail('plan ran a model'))
    report = pilot.main(['--mode', 'long_forward', '--plan_only',
                         '--data_root', str(root), '--checkpoint', str(checkpoint),
                         '--output_dir', str(tmp_path / 'out')])
    from itertools import product
    assert {(c['protein'], c['replica'], c['frames'], c['t'], c['noise_seed'])
            for c in report['cases']} == set(product(['6sup_A'], (1, 2, 3), (4, 250),
                                                     (0.25, 0.5, 0.75), (0, 1, 2)))
    assert len(report['cases']) == 54
    for case in report['cases']:
        cmd = case['command']
        assert '--capacity_only' not in cmd and '--require_gpu' in cmd
        assert cmd[cmd.index('--pdb_id') + 1] == '6sup_A'
        assert cmd[cmd.index('--region_residues') + 1] == '896'
    with pytest.raises(SystemExit):
        pilot.main(['--mode', 'long_forward', '--frames', '64', '--plan_only',
                    '--data_root', str(root), '--checkpoint', str(checkpoint),
                    '--output_dir', str(tmp_path / 'bad')])


def test_ticket3_shards_cover_every_case_once(tmp_path, monkeypatch):
    root, checkpoint = _pilot_inputs(tmp_path, monkeypatch)
    ids = []
    for shard in range(3):
        report = pilot.main(['--mode', 'baseline_forward', '--plan_only',
            '--data_root', str(root), '--checkpoint', str(checkpoint),
            '--output_dir', str(tmp_path / f'out{shard}'),
            '--shard_count', '3', '--shard_index', str(shard)])
        assert len(report['cases']) == 54
        ids.extend(case['id'] for case in report['cases'])
    assert len(set(ids)) == len(ids) == 162


@pytest.mark.parametrize('failure', ['oom', 'missing_gpu', 'timeout', 'program_error'])
def test_ticket3_driver_keeps_execution_failure_classes(tmp_path, monkeypatch, failure):
    import subprocess
    root, checkpoint = _pilot_inputs(tmp_path, monkeypatch)

    def failed(command, stdout, **kwargs):
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(command, 1)
        stdout.write('CUDA out of memory' if failure == 'oom' else failure)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(pilot.subprocess, 'run', failed)
    report = pilot.main(['--mode', 'baseline_forward', '--data_root', str(root),
        '--checkpoint', str(checkpoint), '--output_dir', str(tmp_path / 'out'),
        '--cases', '6sms_A-R1-f4-t0.25-noise0'])
    case, = report['cases']
    assert case['failure_class'] == failure
    assert case['status'] == ('timeout' if failure == 'timeout' else 'failed')
    assert Path(case['log']).is_file()
    assert 'returncode' in case and case['failure_phase']


@pytest.mark.parametrize('payload', [None, '{invalid json'])
def test_ticket3_invalid_child_report_is_a_recorded_program_error(tmp_path, monkeypatch, payload):
    import subprocess
    root, checkpoint = _pilot_inputs(tmp_path, monkeypatch)

    def invalid(command, **kwargs):
        if payload is not None:
            Path(command[command.index('--output') + 1]).write_text(payload)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pilot.subprocess, 'run', invalid)
    report = pilot.main(['--mode', 'baseline_forward', '--data_root', str(root),
        '--checkpoint', str(checkpoint), '--output_dir', str(tmp_path / 'out'),
        '--cases', '6sms_A-R1-f4-t0.25-noise0'])
    case, = report['cases']
    assert report['driver_finished'] and case['failure_class'] == 'program_error'
    assert case['failure_phase'] == 'report_validation'
