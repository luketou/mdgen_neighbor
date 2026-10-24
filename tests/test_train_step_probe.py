"""Seam 1: one real Lightning training step and its JSON report (#9 Ticket 2)."""
import json
import hashlib

import numpy as np

import pytest
import torch

from mdgen.train_step_probe import (batch_shapes, gradient_cosine, main, parameter_group,
                                    run_probe, synthetic_items)

TINY = ['--train_split', 'x', '--val_split', 'x', '--data_dir', 'x', '--num_layers', '2',
        '--embed_dim', '32', '--mha_heads', '4', '--ipa_heads', '2', '--ipa_head_dim', '4',
        '--ipa_qk', '2', '--ipa_v', '2', '--num_frames', '4', '--prepend_ipa',
        '--interleave_ipa', '--sim_condition', '--batch_size', '2']
LOCAL = ['--local_spatial', '--spatial_cell_width', '4', '--spatial_contact_radius', '8',
         '--core_capacity', '4', '--ghost_capacity', '20', '--group_batch_size', '2']


@pytest.fixture(autouse=True)
def restore_precision(monkeypatch, tmp_path):
    monkeypatch.setenv('MODEL_DIR', str(tmp_path / 'model'))
    previous = (torch.get_float32_matmul_precision(),
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    yield
    torch.set_float32_matmul_precision(previous[0])
    torch.backends.cuda.matmul.allow_tf32 = previous[1]
    torch.backends.cudnn.allow_tf32 = previous[2]


def tiny_args(monkeypatch, extra=()):
    from mdgen.parsing import parse_train_args
    monkeypatch.setattr('sys.argv', ['train.py', *TINY, *extra])
    args = parse_train_args()
    delattr(args, 'manual_attention')
    return args


@pytest.mark.parametrize('local', [False, True])
def test_cli_report_and_repeated_fp32_reference_is_deterministic(tmp_path, local):
    output = tmp_path / 'probe' / 'report.json'
    main(['--output', str(output), '--configs', 'fp32-reference', 'fp32-reference',
          'bf16-train', '--synthetic_residues', '12', '--', *TINY, *(LOCAL if local else [])])
    report = json.loads(output.read_text())

    assert report['configs'] == ['fp32-reference', 'fp32-reference', 'bf16-train']
    assert report['batch']['source'] == 'synthetic'
    assert report['batch']['names'] == ['synthetic_0', 'synthetic_1']
    assert report['batch']['shapes']['trans'] == [4, 12, 3]
    assert report['model_args']['local_spatial'] is local
    assert report['model_args']['num_layers'] == 2
    assert report['environment']['torch'] == torch.__version__
    for run, autocast in zip(report['runs'], (None, None, 'torch.bfloat16')):
        state = run['precision_state']
        assert state['autocast_dtype'] == autocast
        assert state['parameter_dtypes'] == ['torch.float32']
        assert run['loss_finite'] and run['loss'] > 0
        assert run['step_time_s'] > 0
        assert run['peak_memory_allocated_bytes'] is None  # CPU
        assert run['gradient_nonfinite'] == 0
        assert run['parameters_nonfinite_after_step'] == 0
        # Disabled cell summaries keep their parameters but receive no gradient.
        assert all('.cell_summaries.' in name for name in run['missing_gradients'])
        assert bool(run['missing_gradients']) is local
        assert 'layers.0.mha_t' in run['gradients']
        assert all(g['norm'] is not None and g['numel'] > 0 for g in run['gradients'].values())
    assert report['runs'][0]['precision_state']['float32_matmul_precision'] == 'highest'
    assert report['runs'][2]['precision_state']['float32_matmul_precision'] == 'high'

    repeat, bf16 = report['comparisons']
    assert set(repeat['gradient_cosine']) == set(report['runs'][0]['gradients'])
    assert all(v == pytest.approx(1, abs=1e-12) for v in repeat['gradient_cosine'].values())
    assert all(v == 0 for v in repeat['max_abs_difference'].values())
    assert repeat['all_groups_pass']
    assert bf16['candidate'] == 'bf16-train'
    assert -1 <= bf16['min_gradient_cosine'] <= 1
    assert bf16['min_gradient_cosine_group'] in bf16['gradient_cosine']
    assert bf16['all_groups_pass'] == all(v >= 0.99 for v in bf16['gradient_cosine'].values())


def test_nonfinite_batch_is_counted(monkeypatch):
    args = tiny_args(monkeypatch)
    items = synthetic_items(args, 12, 2, seed=0)
    items[0]['torsions'][0, 3, 0, 0] = float('nan')
    report = run_probe(args, items, ['fp32-reference', 'fp32-reference'], seed=0)

    run = report['runs'][0]
    assert not run['loss_finite'] and run['loss'] is None
    assert run['gradient_nonfinite'] > 0
    assert any(g['nonfinite'] and g['norm'] is None for g in run['gradients'].values())
    assert all(v is None for v in report['comparisons'][0]['gradient_cosine'].values()
               ) or not report['comparisons'][0]['all_groups_pass']
    json.dumps(report, allow_nan=False)


def test_gradient_cosine_cases():
    a = torch.tensor([1., 2., 0.])
    cosine = gradient_cosine(
        {'same': a, 'opposite': a, 'orthogonal': a, 'zeros': torch.zeros(3),
         'one_zero': a, 'nan': a},
        {'same': a.clone(), 'opposite': -a, 'orthogonal': torch.tensor([2., -1., 5.]),
         'zeros': torch.zeros(3), 'one_zero': torch.zeros(3),
         'nan': torch.tensor([1., float('nan'), 0.])})
    assert cosine['same'] == pytest.approx(1, abs=1e-12)
    assert cosine['opposite'] == pytest.approx(-1, abs=1e-12)
    assert cosine['orthogonal'] == pytest.approx(0, abs=1e-12)
    assert cosine['zeros'] == 1.0
    assert cosine['one_zero'] == 0.0
    assert cosine['nan'] is None


def test_parameter_groups_split_layers_but_not_embedders():
    assert parameter_group('layers.3.mha_t.w_q.weight') == 'layers.3.mha_t'
    assert parameter_group('ipa_layers.0.ipa.linear_q.bias') == 'ipa_layers.0.ipa'
    assert parameter_group('t_embedder.mlp.0.weight') == 't_embedder'
    assert parameter_group('latent_to_emb.weight') == 'latent_to_emb'


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cuda_reports_peak_memory(tmp_path):
    output = tmp_path / 'report.json'
    main(['--output', str(output), '--synthetic_residues', '12', '--', *TINY])
    run = json.loads(output.read_text())['runs'][0]
    assert run['peak_memory_allocated_bytes'] > 0
    assert run['peak_memory_reserved_bytes'] >= run['peak_memory_allocated_bytes']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('local', [False, True])
def test_cuda_bf16_probe_keeps_fp32_weights_and_finite_gradients(tmp_path, local):
    output = tmp_path / 'bf16.json'
    main(['--output', str(output), '--configs', 'bf16-train', '--synthetic_residues', '12',
          '--', *TINY, *(LOCAL if local else [])])
    run = json.loads(output.read_text())['runs'][0]
    assert run['loss_finite']
    assert run['gradient_nonfinite'] == 0
    assert run['parameters_nonfinite_after_step'] == 0
    assert run['precision_state']['parameter_dtypes'] == ['torch.float32']
    assert run['peak_memory_allocated_bytes'] > 0


def test_training_region_report_has_region_diagnostics(tmp_path):
    output = tmp_path / 'region.json'
    region = ['--training_region', '--region_residues', '48']
    main(['--output', str(output), '--configs', 'fp32-reference', 'fp32-reference',
          '--synthetic_residues', '60', '--', *TINY, *LOCAL, *region])
    report = json.loads(output.read_text())

    assert report['batch']['shapes']['trans'] == [4, 48, 3]
    assert report['batch']['shapes']['loss_core_mask'] == [48]
    regions = report['batch']['regions']
    assert len(regions) == 2
    for entry in regions:
        assert 0 < entry['loss_residues'] < entry['residues'] <= 48
        assert entry['loss_groups'] >= 1 and entry['halo_groups'] >= 1
        assert entry['next_region_size'] is None or entry['next_region_size'] > 48
    assert all(run['loss_finite'] and run['gradient_nonfinite'] == 0 for run in report['runs'])
    repeat = report['comparisons'][0]
    assert all(v == 0 for v in repeat['max_abs_difference'].values())
    for run in report['runs']:  # the collated loader batch, checked on every run
        shapes = run['batch_shapes']
        assert shapes['trans'] == [2, 4, 48, 3] and shapes['loss_core_mask'] == [2, 48]
        assert shapes['partition.residue_mask'] == [2, 48]
        assert all(shape[0] == 2 for shape in shapes.values())

    again = tmp_path / 'again.json'
    main(['--output', str(again), '--synthetic_residues', '60', '--', *TINY, *LOCAL, *region])
    assert json.loads(again.read_text())['batch']['regions'] == regions  # seeded by --seed


def test_batch_shapes_rejects_a_tensor_without_the_batch_dimension():
    batch = {'trans': torch.zeros(2, 4, 3), 'name': ['a', 'b'],
             'region': {'center': torch.zeros(2, dtype=torch.long)}}
    assert batch_shapes(batch, 2) == {'trans': [2, 4, 3], 'region.center': [2]}
    with pytest.raises(ValueError, match='region.center'):
        batch_shapes({**batch, 'region': {'center': torch.zeros(3)}}, 2)
    with pytest.raises(ValueError, match='name'):
        batch_shapes({**batch, 'name': ['a']}, 2)


def test_cli_real_replica_uses_checkpoint_weights_and_first_frame(monkeypatch, tmp_path):
    from mdgen.wrapper import NewMDGenWrapper
    args = tiny_args(monkeypatch)
    model = NewMDGenWrapper(args)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint = tmp_path / 'zero.ckpt'
    torch.save({'state_dict': model.state_dict(), 'hyper_parameters': {'args': args}}, checkpoint)
    atoms = np.random.default_rng(0).normal(size=(5, 12, 14, 3)).astype(np.float32)
    np.save(tmp_path / 'prot_R1.npy', atoms)
    split = tmp_path / 'split.csv'
    split.write_text('name,seqres\nprot,ACDEFGHIKLMN\n')
    output = tmp_path / 'real.json'
    main(['--output', str(output), '--sim_ckpt', str(checkpoint), '--time', '0.5', '--', *TINY,
          '--data_dir', str(tmp_path), '--train_split', str(split), '--val_split', str(split),
          '--batch_size', '1', '--atlas', '--replicates', '1', '--crop', '12', '--overfit_frame'])
    report = json.loads(output.read_text())
    assert report['checkpoint']['sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert report['runs'][0]['initial_parameter_norm'] == 0
    assert report['batch']['names'] == ['prot_R1']
    assert report['batch']['frame_starts'] == [0]
    assert report['runs'][0]['loss_finite']
    assert report['runs'][0]['training_time'] == 0.5


@pytest.mark.parametrize('crop', ['8', '16'])
def test_full_protein_mode_rejects_a_crop_that_differs_from_the_protein(tmp_path, crop):
    np.save(tmp_path / 'prot_R1.npy',
            np.random.default_rng(0).normal(size=(5, 12, 14, 3)).astype(np.float32))
    split = tmp_path / 'split.csv'
    split.write_text('name,seqres\nprot,ACDEFGHIKLMN\n')
    with pytest.raises(ValueError, match=f'--crop {crop} differs from prot length 12'):
        main(['--output', str(tmp_path / 'cropped.json'), '--time', '0.5', '--', *TINY,
              '--data_dir', str(tmp_path), '--train_split', str(split), '--val_split', str(split),
              '--batch_size', '1', '--atlas', '--replicates', '1', '--crop', crop,
              '--overfit_frame'])
    assert not (tmp_path / 'cropped.json').exists()


def test_full_protein_crop_check_uses_the_row_overfit_actually_loads(tmp_path):
    np.save(tmp_path / 'prot_R1.npy',
            np.random.default_rng(0).normal(size=(5, 12, 14, 3)).astype(np.float32))
    split = tmp_path / 'split.csv'
    split.write_text('name,seqres\nprot,ACDEFGHIKLMN\nshort,ACDEFGHI\n')
    output = tmp_path / 'overfit.json'
    main(['--output', str(output), '--time', '0.5', '--dataset_index', '1', '--', *TINY,
          '--data_dir', str(tmp_path), '--train_split', str(split), '--val_split', str(split),
          '--batch_size', '1', '--atlas', '--replicates', '1', '--crop', '12', '--overfit',
          '--overfit_frame'])
    assert json.loads(output.read_text())['batch']['names'] == ['prot_R1']


def test_cli_training_region_accepts_an_explicit_candidate_group(tmp_path):
    output = tmp_path / 'fixed.json'
    main(['--output', str(output), '--synthetic_residues', '60', '--seed_group', '0',
          '--', *TINY, *LOCAL, '--training_region', '--region_residues', '60'])
    report = json.loads(output.read_text())
    for region in report['batch']['regions']:
        assert region['seed_group'] == 0
        assert 0 in region['loss_group_ids']
        assert len(region['loss_group_ids']) == region['loss_groups']


def test_cli_full_coverage_control_shares_noise_and_supervision(tmp_path):
    output = tmp_path / 'control.json'
    main(['--output', str(output), '--coverage_control', '--synthetic_residues', '12',
          '--time', '0.5', '--', *TINY, *LOCAL, '--batch_size', '1',
          '--region_residues', '20', '--halo_rings', '1'])
    report = json.loads(output.read_text())
    assert report['probe'] == 'full-coverage-control'
    assert report['thresholds_applied'] is False
    alignment = report['alignment']
    assert alignment['full_noise_sha256'] == alignment['region_noise_sha256']
    assert alignment['full_mask_sha256'] == alignment['region_mask_sha256']
    assert alignment['full_denominator'] == alignment['region_denominator'] > 0
    assert [r['batch_shapes']['trans'][2] for r in report['runs']] == [12, 12, 20]
    repeat, coverage = report['comparisons']
    assert repeat['kind'] == 'same_path_repeat'
    assert all(g['difference_rms'] == 0 for g in repeat['gradients'].values())
    assert all(g['difference_rms'] == 0 for g in repeat['updates'].values())
    assert coverage['kind'] == 'full_coverage'
    assert max(g['difference_rms'] for g in coverage['gradients'].values()) < 1e-5
    assert any(g['reference_norm'] > 0 for g in coverage['updates'].values())
    zero_gradients = [g for g in coverage['gradients'].values() if g['reference_zero']]
    assert zero_gradients and all(g['cosine'] is None for g in zero_gradients)


def test_control_without_time_still_uses_the_same_actual_time(tmp_path):
    output = tmp_path / 'default-time.json'
    main(['--output', str(output), '--coverage_control', '--synthetic_residues', '12',
          '--', *TINY, *LOCAL, '--batch_size', '1', '--region_residues', '20'])
    report = json.loads(output.read_text())
    assert [r['training_time'] for r in report['runs']] == [0.5, 0.5, 0.5]


def test_seed_group_without_a_region_mode_names_both_region_options(tmp_path):
    with pytest.raises(ValueError, match='--training_region or --coverage_control'):
        main(['--output', str(tmp_path / 'invalid.json'), '--seed_group', '0',
              '--synthetic_residues', '60', '--', *TINY, *LOCAL, '--batch_size', '1'])
    assert not (tmp_path / 'invalid.json').exists()


def test_control_rejects_a_truncated_region_before_running_steps(tmp_path):
    with pytest.raises(ValueError, match='every full-protein residue'):
        main(['--output', str(tmp_path / 'invalid.json'), '--coverage_control',
              '--synthetic_residues', '60', '--', *TINY, *LOCAL, '--batch_size', '1',
              '--region_residues', '40', '--halo_rings', '0'])
    assert not (tmp_path / 'invalid.json').exists()


def test_real_control_keeps_the_original_random_replica_and_window(tmp_path):
    atoms = np.random.default_rng(1).normal(size=(9, 12, 14, 3)).astype(np.float32)
    for replica in (1, 2):
        np.save(tmp_path / f'prot_R{replica}.npy', atoms * replica)
    split = tmp_path / 'split.csv'
    split.write_text('name,seqres\nprot,ACDEFGHIKLMN\n')
    output = tmp_path / 'real-control.json'
    main(['--output', str(output), '--coverage_control', '--', *TINY, *LOCAL,
          '--data_dir', str(tmp_path), '--train_split', str(split), '--val_split', str(split),
          '--batch_size', '1', '--atlas', '--replicates', '1', '2', '--crop', '12',
          '--region_residues', '20'])
    report = json.loads(output.read_text())
    assert report['alignment']['name'] in ('prot_R1', 'prot_R2')
    assert 0 <= report['alignment']['frame_start'] <= 5
    assert report['alignment']['full_mask_sha256'] == report['alignment']['region_mask_sha256']
    assert max(g['difference_rms'] for g in report['comparisons'][1]['gradients'].values()) < 1e-5


def test_control_identifies_nonfinite_predictions_separately_from_inputs(monkeypatch, tmp_path):
    from mdgen.wrapper import NewMDGenWrapper
    args = tiny_args(monkeypatch, LOCAL)
    model = NewMDGenWrapper(args)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float('nan'))
    checkpoint = tmp_path / 'nonfinite.ckpt'
    torch.save({'state_dict': model.state_dict(), 'hyper_parameters': {'args': args}}, checkpoint)
    output = tmp_path / 'nonfinite-control.json'
    main(['--output', str(output), '--coverage_control', '--sim_ckpt', str(checkpoint),
          '--synthetic_residues', '12', '--', *TINY, *LOCAL,
          '--batch_size', '1', '--region_residues', '20'])
    report = json.loads(output.read_text())
    for run in report['runs']:
        assert run['nonfinite_counts']['latents'] == run['nonfinite_counts']['noise'] == 0
        assert run['nonfinite_counts']['model_output'] > 0
        assert run['gradient_nonfinite'] > 0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('residues,budget,halo,expected', [
    (12, 20, 1, 'passed'), (60, 48, 1, 'descriptive')])
def test_paired_region_uses_same_objective_and_real_optimizer(tmp_path, residues, budget, halo, expected):
    output = tmp_path / 'paired.json'
    report = main(['--output', str(output), '--paired_region', '--synthetic_residues', str(residues),
        '--seed_group', '0', '--time', '0.5', '--warmup', '0', '--repeats', '1', '--',
        *TINY, *LOCAL, '--batch_size', '1', '--region_residues', str(budget),
        '--halo_rings', str(halo), '--grad_checkpointing'])
    assert report['status'] == expected
    alignment = report['alignment']
    assert alignment['full_denominator'] == alignment['region_denominator'] > 0
    assert alignment['full_noise_sha256'] == alignment['region_noise_sha256']
    assert alignment['full_mask_sha256'] == alignment['region_mask_sha256']
    measured = report['measurements'][0]
    assert measured['full']['initial_state_sha256'] == measured['candidate']['initial_state_sha256']
    for side in ('full', 'candidate'):
        run = measured[side]
        assert run['parameters_nonfinite_after_step'] == 0
        assert run['loss_components']['denominator'] == alignment['full_denominator']
        assert run['loss_components']['numerator'] / run['loss_components']['denominator'] == pytest.approx(run['loss'])
        saved = torch.load(run['artifact_path'], weights_only=True)
        assert saved['pre_clip_gradients'] and saved['updates']
    if expected == 'descriptive':
        assert measured['comparison']['control_passed'] is None
        assert len(alignment['region_residue_ids']) < residues
        assert len(alignment['loss_residue_ids']) < residues
    else:
        assert measured['comparison']['control_passed']
    json.dumps(report, allow_nan=False)
    with pytest.raises(FileExistsError):
        main(['--output', str(output), '--paired_region'])


def test_paired_denominators_reject_unequal_and_zero_is_not_failure():
    from mdgen.train_step_control import check_denominators
    with pytest.raises(ValueError, match='denominators differ'):
        check_denominators(torch.ones(2), torch.ones(3))
    assert check_denominators(torch.zeros(2), torch.zeros(3))['status'] == 'no_supervision'
    with pytest.raises(ValueError, match='invalid'):
        check_denominators(torch.tensor([float('nan')]), torch.ones(1))


def test_paired_zero_supervision_skips_backward(monkeypatch, tmp_path):
    import mdgen.train_step_control as control
    args = tiny_args(monkeypatch, [*LOCAL, '--batch_size', '1', '--region_residues', '20',
                                   '--cond_interval', '1'])
    args.region_seed_group, args.probe_time = 0, 0.5
    def forbidden(*args, **kwargs):
        pytest.fail('zero supervision must not run a training step')
    monkeypatch.setattr(control, 'run_step', forbidden)
    items = synthetic_items(args, 12, 1, 0)
    items[0]['torsions'][0, 0, 0, 0] = float('nan')
    report = control.run_paired_region(args, items, 0, synthetic=True, artifact_dir=tmp_path)
    assert report['status'] == 'no_supervision' and report['measurements'] == []


def test_halo_change_keeps_loss_group_and_mask(monkeypatch):
    from mdgen.train_step_control import prepare_paired_items
    args = tiny_args(monkeypatch, [*LOCAL, '--batch_size', '1', '--region_residues', '60'])
    args.region_seed_group, args.probe_time = 0, 0.5
    item = synthetic_items(args, 80, 1, 0)[0]
    args.halo_rings = 1
    _, _, a = prepare_paired_items(args, item, 0, synthetic=True, use_sdpa=True)
    args.halo_rings = 2
    _, _, b = prepare_paired_items(args, item, 0, synthetic=True, use_sdpa=True)
    assert a['paired_common_subset'] and b['paired_common_subset']
    assert a['loss_group_ids'] == b['loss_group_ids']
    assert a['loss_residue_ids'] == b['loss_residue_ids']
    assert a['full_denominator'] == b['full_denominator']


def test_paired_nonfinite_fails_even_when_truncated(monkeypatch):
    from mdgen.train_step_control import compare_paired_runs
    run = dict(loss=1., loss_finite=True, gradient_nonfinite=0,
               parameters_nonfinite_after_step=0, missing_gradients=[],
               initial_state_sha256='same', optimizer_initial_state='fresh_empty', optimizer={})
    a, b = {'group': torch.ones(3)}, {'group': torch.tensor([float('nan'), 1., 1.])}
    comparison = compare_paired_runs(run, run, a, b, a, a, control=False)
    assert comparison['status'] == 'failed' and comparison['control_passed'] is None
    assert comparison['gradients']['group']['nonfinite'] == 1
    json.dumps(comparison, allow_nan=False)


def test_paired_oom_evidence_keeps_full_backward_stage(monkeypatch, tmp_path):
    import mdgen.train_step_control as control
    from mdgen.train_step_matrix import failure_category
    args = tiny_args(monkeypatch, [*LOCAL, '--batch_size', '1', '--region_residues', '20'])
    args.region_seed_group, args.probe_time = 0, .5
    def oom(*args, **kwargs):
        error = torch.OutOfMemoryError('CUDA out of memory')
        error.probe_evidence = {'stage': 'backward', 'peak_memory_allocated_bytes': 123}
        raise error
    monkeypatch.setattr(control, 'run_step', oom)
    with pytest.raises(torch.OutOfMemoryError) as caught:
        control.run_paired_region(args, synthetic_items(args, 12, 1, 0), 0, synthetic=True)
    assert caught.value.probe_evidence['stage'] == 'backward'
    assert caught.value.probe_evidence['side'] == 'full'
    assert failure_category(caught.value) == 'oom'


def test_paired_control_enforces_relative_option_b_and_near_zero_policy():
    from mdgen.train_step_control import compare_paired_runs
    run = dict(loss=1., loss_finite=True, gradient_nonfinite=0,
               parameters_nonfinite_after_step=0, missing_gradients=[],
               initial_state_sha256='same', optimizer_initial_state='fresh_empty', optimizer={})
    gradient = {'active': torch.full((4,), 1e-5), 'zero': torch.zeros(4)}
    changed = {**gradient, 'active': gradient['active'] * 1.001}
    update = {'active': torch.full((4,), 1e-3), 'zero': torch.zeros(4)}
    comparison = compare_paired_runs(run, run, gradient, changed, update, update, control=True)
    assert comparison['status'] == 'failed'
    assert comparison['gradients']['active']['difference_rms'] < 1e-7
    assert comparison['gradients']['active']['relative_difference'] > 5e-4
    assert comparison['gradients']['zero']['cosine'] is None
    assert comparison['gradients']['zero']['near_zero_reference']


def test_paired_missing_gpu_writes_failure_evidence(monkeypatch, tmp_path):
    monkeypatch.setenv('MDGEN_PAIRED_REQUIRE_GPU', '1')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='missing_gpu'):
        main(['--paired_region', '--output', str(tmp_path / 'probe.json')])
    failure = json.loads((tmp_path / 'failure.json').read_text())
    assert failure['failure_category'] == 'missing_gpu'
    assert 'missing_gpu' in failure['traceback']


def test_actual_backward_oom_callback_records_stage_and_traceback(monkeypatch, tmp_path):
    from mdgen.wrapper import NewMDGenWrapper
    original = NewMDGenWrapper.training_step
    def fail_backward(self, batch, batch_idx):
        loss = original(self, batch, batch_idx)
        def oom(gradient):
            raise torch.OutOfMemoryError('CUDA out of memory: CPU 測試注入')
        loss.register_hook(oom)
        return loss
    monkeypatch.setattr(NewMDGenWrapper, 'training_step', fail_backward)
    with pytest.raises(torch.OutOfMemoryError):
        main(['--paired_region', '--output', str(tmp_path / 'probe.json'),
              '--synthetic_residues', '12', '--seed_group', '0', '--time', '.5',
              '--warmup', '0', '--repeats', '1', '--', *TINY, *LOCAL,
              '--batch_size', '1', '--region_residues', '20'])
    evidence = json.loads((tmp_path / 'failure.json').read_text())
    assert evidence['stage'] == 'backward' and evidence['side'] == 'full'
    assert evidence['failure_category'] == 'oom'
    assert evidence['alignment']['full_denominator'] > 0
    assert 'CPU 測試注入' in evidence['traceback']
