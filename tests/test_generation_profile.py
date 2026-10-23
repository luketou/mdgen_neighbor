"""Full-length generation profile (#9 Ticket 8), on a tiny random model on CPU."""
import json

import numpy as np
import pytest
import torch

from mdgen.full_length.generation_profile import kernel_stats, main

T = 3
TINY = ['--train_split', 'x', '--val_split', 'x', '--data_dir', 'x', '--num_layers', '2',
        '--embed_dim', '32', '--mha_heads', '4', '--ipa_heads', '2', '--ipa_head_dim', '4',
        '--ipa_qk', '2', '--ipa_v', '2', '--num_frames', str(T), '--prepend_ipa',
        '--interleave_ipa', '--sim_condition', '--local_spatial', '--spatial_cell_width', '8',
        '--spatial_contact_radius', '8', '--core_capacity', '8', '--ghost_capacity', '32',
        '--group_batch_size', '2']


@pytest.fixture
def checkpoint(monkeypatch, tmp_path):
    from mdgen.parsing import parse_train_args
    from mdgen.wrapper import NewMDGenWrapper
    monkeypatch.setenv('MODEL_DIR', str(tmp_path / 'model'))
    monkeypatch.setattr('sys.argv', ['train.py', *TINY])
    args = parse_train_args()
    delattr(args, 'manual_attention')
    torch.manual_seed(0)
    wrapper = NewMDGenWrapper(args)
    with torch.no_grad():
        for p in wrapper.model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    path = tmp_path / 'tiny.ckpt'
    torch.save({'hyper_parameters': {'args': args}, 'state_dict': wrapper.state_dict()}, path)
    return path


def test_report_counts_nfe_and_profiles_one_forward(tmp_path, checkpoint):
    output = tmp_path / 'profile.json'
    main(['--sim_ckpt', str(checkpoint), '--output', str(output), '--synthetic_residues', '40',
          '--rollouts', '2', '--ode_atol', '1e-3', '--ode_rtol', '1e-2'])
    report = json.loads(output.read_text())
    assert report['residues'] == 40
    runs = report['generation']['rollouts']
    assert len(runs) == 2
    assert all(run['nfe'] > 0 and run['wall_clock_s'] > 0 for run in runs)
    assert report['generation']['nfe_median'] == np.median([run['nfe'] for run in runs])
    partition = report['partition']
    assert partition['groups'] > 1 and partition['group_batch_size'] == 2
    # --prepend_ipa adds an IPA layer per main layer; each of the 4 layers runs local
    # IPA and local attention (--interleave_ipa), and each call walks every group chunk
    assert partition['spatial_calls_per_forward'] == 8
    assert partition['group_chunks_per_forward'] == 8 * partition['chunks_per_spatial_call']
    assert report['generation']['sampling_method'] == 'dopri5'
    profile = report['forward_profile']
    regions = profile['regions']
    # every layer's spatial hooks were annotated once per forward
    assert regions['mdgen.local_attention']['calls'] == 4
    assert regions['mdgen.local_ipa']['calls'] == 4
    assert report['forward_s'] > 0 and report['settings']['model']['num_frames'] == T


def test_tighter_tolerance_needs_more_evaluations(tmp_path, checkpoint):
    nfe = []
    for atol in ('1e-2', '1e-4'):
        output = tmp_path / f'{atol}.json'
        main(['--sim_ckpt', str(checkpoint), '--output', str(output), '--synthetic_residues',
              '30', '--ode_atol', atol, '--ode_rtol', atol, '--no_profile'])
        report = json.loads(output.read_text())
        assert 'forward_profile' not in report
        nfe.append(report['generation']['rollouts'][0]['nfe'])
    assert nfe[1] > nfe[0]


def test_kernel_stats_measure_gpu_idle_time_and_region_launches():
    events = [
        ('mdgen.local_attention', 'cpu', 0, 50),
        ('cudaLaunchKernel', 'cpu', 5, 6), ('cudaLaunchKernelExC_v11060', 'cpu', 20, 21),
        ('cudaLaunchKernel', 'cpu', 60, 61), ('aten::mm', 'cpu', 59, 62),
        ('k1', 'cuda', 10, 14), ('k2', 'cuda', 12, 16), ('k3', 'cuda', 90, 110),
        ('Memset (Device)', 'cuda', 30, 31),
    ]
    stats = kernel_stats(events)
    assert stats['kernel_count'] == 3 and stats['launch_count'] == 3
    assert stats['memory_op_count'] == 1
    assert stats['kernel_span_us'] == 100 and stats['gpu_busy_us'] == 26  # overlap counted once
    assert stats['gpu_busy_fraction'] == pytest.approx(0.26)
    assert stats['kernel_fraction_under_10us'] == pytest.approx(2 / 3)
    attention = stats['regions']['mdgen.local_attention']
    assert attention['calls'] == 1 and attention['launch_count'] == 2
    assert attention['launch_fraction'] == pytest.approx(2 / 3)
    assert stats['regions']['mdgen.local_ipa']['launch_count'] == 0


def test_protein_source_reads_frame_zero(tmp_path, checkpoint):
    rng = np.random.default_rng(0)
    L = 40
    step = rng.normal(size=(L, 3)) + [0.5, 0, 0]
    ca = (3.8 * step / np.linalg.norm(step, axis=-1, keepdims=True)).cumsum(0)
    atoms = ca[None, :, None] + rng.normal(size=(2, L, 14, 3))
    np.save(tmp_path / 'prot_R1.npy', atoms.astype(np.float32))
    (tmp_path / 'split.csv').write_text(f'name,seqres\nprot,{"ACDEFGHIKLMNPQRSTVWY" * 2}\n')
    output = tmp_path / 'protein.json'
    main(['--sim_ckpt', str(checkpoint), '--output', str(output), '--pdb_id', 'prot',
          '--suffix', '_R1', '--data_dir', str(tmp_path), '--split', str(tmp_path / 'split.csv'),
          '--ode_atol', '1e-3', '--ode_rtol', '1e-2', '--no_profile'])
    assert json.loads(output.read_text())['residues'] == L
    with pytest.raises(SystemExit):
        main(['--sim_ckpt', str(checkpoint), '--output', str(output), '--pdb_id', 'typo',
              '--data_dir', str(tmp_path), '--split', str(tmp_path / 'split.csv')])
