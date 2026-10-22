"""Seam 1: one real Lightning training step and its JSON report (#9 Ticket 2)."""
import json

import pytest
import torch

from mdgen.train_step_probe import gradient_cosine, main, parameter_group, run_probe, synthetic_items

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cuda_reports_peak_memory(tmp_path):
    output = tmp_path / 'report.json'
    main(['--output', str(output), '--synthetic_residues', '12', '--', *TINY])
    run = json.loads(output.read_text())['runs'][0]
    assert run['peak_memory_allocated_bytes'] > 0
    assert run['peak_memory_reserved_bytes'] >= run['peak_memory_allocated_bytes']


def test_training_region_report_has_region_diagnostics(tmp_path):
    output = tmp_path / 'region.json'
    region = ['--training_region', '--region_core', '8', '--region_ghost', '16']
    main(['--output', str(output), '--configs', 'fp32-reference', 'fp32-reference',
          '--synthetic_residues', '60', '--', *TINY, *LOCAL, *region])
    report = json.loads(output.read_text())

    assert report['batch']['shapes']['trans'] == [4, 24, 3]
    assert report['batch']['shapes']['core_mask'] == [24]
    regions = report['batch']['regions']
    assert len(regions) == 2
    for entry in regions:
        assert 0 <= entry['center'] < 60
        assert entry['core_count'] == 8 and 0 < entry['ghost_count'] <= 16
        assert entry['spatial_truncated_count'] >= 0
        assert entry['mandatory_retained_count'] == entry['mandatory_neighbor_count']
    assert all(run['loss_finite'] and run['gradient_nonfinite'] == 0 for run in report['runs'])
    repeat = report['comparisons'][0]
    assert all(v == 0 for v in repeat['max_abs_difference'].values())

    again = tmp_path / 'again.json'
    main(['--output', str(again), '--synthetic_residues', '60', '--', *TINY, *LOCAL, *region])
    assert json.loads(again.read_text())['batch']['regions'] == regions  # seeded by --seed
