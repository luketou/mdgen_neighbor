"""Halo approximation error experiment (#12 Ticket 5), on a tiny random model."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mdgen.full_length.group_region import sample_region_groups, with_halo
from mdgen.full_length.halo_error import boundary_rings, halo_errors, main, protein_dataset
from mdgen.full_length.partition import LocalGroup, SpatialPartition

T, L = 3, 200
LOCAL = ['--local_spatial', '--spatial_cell_width', '8', '--spatial_contact_radius', '8',
         '--core_capacity', '8', '--ghost_capacity', '32']
TINY = ['--train_split', 'x', '--val_split', 'x', '--data_dir', 'x', '--num_layers', '1',
        '--embed_dim', '32', '--mha_heads', '4', '--ipa_heads', '2', '--ipa_head_dim', '4',
        '--ipa_qk', '2', '--ipa_v', '2', '--num_frames', str(T), '--prepend_ipa',
        '--interleave_ipa', '--sim_condition', *LOCAL]


def write_protein(tmp_path):
    rng = np.random.default_rng(0)
    step = rng.normal(size=(L, 3)) + [0.5, 0, 0]
    ca = (3.8 * step / np.linalg.norm(step, axis=-1, keepdims=True)).cumsum(0)
    atoms = ca[None, :, None] + rng.normal(size=(T, L, 14, 3))
    atoms[:, :, 1] = ca + rng.normal(scale=0.3, size=(T, L, 3))
    np.save(tmp_path / 'prot_R1.npy', atoms.astype(np.float32))
    split = tmp_path / 'split.csv'
    split.write_text(f'name,seqres\nprot,{"ACDEFGHIKLMNPQRSTVWY" * (L // 20)}\n')
    return split


@pytest.fixture
def wrapper(monkeypatch, tmp_path):
    from mdgen.parsing import parse_train_args
    from mdgen.wrapper import NewMDGenWrapper
    monkeypatch.setenv('MODEL_DIR', str(tmp_path / 'model'))
    monkeypatch.setattr('sys.argv', ['train.py', *TINY])
    args = parse_train_args()
    delattr(args, 'manual_attention')
    torch.manual_seed(0)
    wrapper = NewMDGenWrapper(args)
    with torch.no_grad():  # move zero-initialised heads so outputs depend on context
        for p in wrapper.model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return wrapper.eval()


@pytest.fixture
def checkpoint(wrapper, tmp_path):
    path = tmp_path / 'tiny.ckpt'
    torch.save({'hyper_parameters': {'args': wrapper.args}, 'state_dict': wrapper.state_dict()}, path)
    return path


def test_in_memory_halo_comparison_keeps_the_same_loss_residues(tmp_path, wrapper):
    from mdgen.dataset import collate

    split = write_protein(tmp_path)
    dataset = protein_dataset(SimpleNamespace(data_dir=str(tmp_path), suffix='',
                              partition_dir=None, split=split, replicate=1), wrapper.args)
    item = dataset[0]
    arr = np.load(tmp_path / 'prot_R1.npy', mmap_mode='r')
    partition, ca, _ = dataset.full_partition('prot_R1', arr)
    result = halo_errors(wrapper, collate([item]), partition, ca,
                         region_residues=80, halo_rings=(0, 1), regions=2,
                         times=(0.3, 0.7), seed=0)

    loss_count = 0
    for region in result['regions']:
        shallow, deep = region['by_halo']['0'], region['by_halo']['1']
        assert shallow['loss_residues'] == deep['loss_residues'] > 0
        assert shallow['region_residues'] <= deep['region_residues'] <= 80
        loss_count += shallow['loss_residues']
    for rows in result['residues'].values():
        assert len(rows) == 2 * loss_count
        assert np.isfinite([value for _, l2, relative in rows for value in (l2, relative)]).all()
    assert max(relative for _, _, relative in result['residues'][0]) > 1e-3


def run(tmp_path, checkpoint, *extra):
    split = write_protein(tmp_path)
    output = tmp_path / 'report.json'
    main(['--sim_ckpt', str(checkpoint), '--data_dir', str(tmp_path), '--split', str(split),
          '--replicate', '1', '--output', str(output), '--region_residues', '80',
          '--regions', '2', '--times', '0.3', '0.7', *LOCAL[1:], *extra])
    return json.loads(output.read_text())


@pytest.mark.checkpoint
def test_report_compares_halo_settings_on_the_same_loss_groups(tmp_path, checkpoint):
    report = run(tmp_path, checkpoint, '--halo_rings', '0', '1', '2')
    protein, = report['proteins']
    assert protein['name'] == 'prot_R1' and protein['residues'] == L
    loss_residues = 0
    for region in protein['regions']:
        sizes = [region['by_halo'][k]['region_residues'] for k in ('0', '1', '2')]
        assert sizes == sorted(sizes) and sizes[-1] <= 80  # deeper halo, same loss groups
        losses = {region['by_halo'][k]['loss_residues'] for k in ('0', '1', '2')}
        assert len(losses) == 1
        loss_residues += losses.pop()
        assert all(len(region['by_halo'][k]['times']) == 2 for k in ('0', '1', '2'))
    summary = report['summary']
    for k in ('0', '1', '2'):
        assert summary[k]['relative']['count'] == 2 * loss_residues  # two times per region
        assert summary[k]['relative']['max'] >= summary[k]['relative']['median'] >= 0
        assert '0' in summary[k]['by_ring']
    # without a halo, loss cores near the boundary miss context and their outputs differ
    assert summary['0']['by_ring']['0']['relative']['max'] > 1e-3


@pytest.mark.checkpoint
def test_a_halo_covering_the_protein_reproduces_the_full_forward(monkeypatch, tmp_path, checkpoint):
    def one_loss_group(partition, ca, generator, **kwargs):  # keep a small loss set
        return with_halo(partition, (int(torch.randint(len(partition.groups), (),
                                                         generator=generator)),), 0)
    monkeypatch.setattr('mdgen.full_length.halo_error.sample_region_groups', one_loss_group)
    report = run(tmp_path, checkpoint, '--halo_rings', '0', '200')
    region = report['proteins'][0]['regions'][0]
    assert region['by_halo']['200']['region_residues'] == L
    assert region['by_halo']['0']['region_residues'] < L
    assert report['summary']['200']['l2']['max'] < 1e-4
    assert report['summary']['0']['l2']['max'] > 1e-3


@pytest.mark.checkpoint
def test_region_budget_counts_the_deepest_halo(tmp_path, checkpoint):
    with pytest.raises(ValueError, match='region_residues'):
        run(tmp_path, checkpoint, '--halo_rings', '1', '200')


def test_seeds_that_do_not_fit_are_skipped_and_reported(monkeypatch, tmp_path, checkpoint):
    calls = []

    def first_seed_too_big(*args, **kwargs):
        calls.append(None)
        if len(calls) == 1:
            raise ValueError('region_residues=80 cannot hold seed group 0')
        return sample_region_groups(*args, **kwargs)
    monkeypatch.setattr('mdgen.full_length.halo_error.sample_region_groups', first_seed_too_big)
    protein, = run(tmp_path, checkpoint, '--halo_rings', '0', '1')['proteins']
    assert protein['skipped_seeds'] == [{'seed': 0, 'reason': 'region_residues=80 cannot hold seed group 0'}]
    assert len(protein['regions']) == 1


def test_boundary_rings_count_hops_to_groups_outside_the_loss_set():
    # chain of groups 0-1-2-3: each group's ghost is a core of its neighbours
    cores = [[0], [1], [2], [3]]
    ghosts = [[1], [0, 2], [1, 3], [2]]
    groups = tuple(LocalGroup((i, 0, 0), torch.tensor(c), torch.tensor(g))
                   for i, (c, g) in enumerate(zip(cores, ghosts)))
    partition = SpatialPartition(groups, torch.arange(4), torch.ones(4, dtype=torch.bool),
                                 1, 2, 8.0)
    assert boundary_rings(partition, [0, 1, 2]) == {0: 2, 1: 1, 2: 0}
    assert boundary_rings(partition, [1, 2, 3]) == {1: 0, 2: 1, 3: 2}
    assert boundary_rings(partition, [0, 1, 2, 3]) == {g: None for g in range(4)}


def test_with_halo_keeps_the_loss_groups():
    # built here: on HPC an installed `tests` package can shadow `tests.test_group_region`
    from mdgen.full_length.partition import build_spatial_partition
    step = torch.randn(120, 3, generator=torch.Generator().manual_seed(0))
    ca = (3.8 * step / step.norm(dim=-1, keepdim=True)).cumsum(0)
    partition = build_spatial_partition(
        ca, mask=torch.ones(120, dtype=torch.bool), chain_id=(torch.arange(120) >= 60).long(),
        residue_index=torch.arange(120) % 60, cell_width=8.0, contact_radius=8.0,
        core_capacity=8, ghost_capacity=64)
    selection = sample_region_groups(partition, ca, region_residues=90, halo_rings=2,
                                     generator=torch.Generator().manual_seed(0))
    for k in (0, 1, 2):
        again = with_halo(partition, selection.loss_groups, k)
        assert again.loss_groups == selection.loss_groups
    assert torch.equal(with_halo(partition, selection.loss_groups, 2).residues, selection.residues)


def test_unknown_pdb_id_is_rejected_before_loading(tmp_path):
    split = write_protein(tmp_path)
    with pytest.raises(SystemExit):
        main(['--sim_ckpt', str(tmp_path / 'missing.ckpt'), '--data_dir', str(tmp_path),
              '--split', str(split), '--output', str(tmp_path / 'r.json'), '--pdb_id', 'typo'])


def test_cli_short_window_reports_precision_and_tail_statistics(tmp_path, checkpoint):
    report = run(tmp_path, checkpoint, '--halo_rings', '0', '1', '--num_frames', '2',
                 '--warmup', '0', '--repeats', '1')
    assert report['settings']['model']['num_frames'] == 2
    assert report['precision_state']['float32_matmul_precision'] == 'highest'
    assert not report['precision_state']['cuda_matmul_allow_tf32']
    protein, = report['proteins']
    assert protein['batch_shapes']['trans'] == [1, 2, L, 3]
    # Every input tensor is recorded with its batch dimension, not only `trans`.
    assert {'rots', 'torsions', 'seqres', 'mask'} <= set(protein['batch_shapes'])
    assert all(shape[0] == 1 for shape in protein['batch_shapes'].values())
    assert protein['summary']['0']['l2']['p95'] >= protein['summary']['0']['l2']['median']
    assert len(protein['measurements']['full']) == 2  # one per t


def test_cli_fixed_candidates_preserves_duplicates_and_all_group_coverage(tmp_path, checkpoint):
    report = run(tmp_path, checkpoint, '--halo_rings', '0', '1',
                 '--region_residues', str(L), '--seed_groups', '0', '0', '1')
    protein, = report['proteins']
    assert protein['coverage']['attempted'] == 3
    assert protein['coverage']['feasible'] == 3
    assert protein['coverage']['duplicates'] == 2
    assert protein['coverage']['unique_regions'] == 1
    assert protein['coverage']['loss_residue_fraction'] == 1.0
    assert len(protein['seed_regions']['1']['sizes']) == protein['groups']
    region, = protein['regions']
    assert region['by_halo']['1']['truncated'] is False
    sample = protein['observations'][0]
    assert sample['effective_components'] > 0
    assert sample['reference_class'] == 'regular'
    assert sample['relative'] == 0
    assert protein['strata'][0]['l2']['p95'] == 0


def test_partition_inventory_never_runs_a_model_and_keeps_infeasible_groups(tmp_path, checkpoint):
    report = run(tmp_path, checkpoint, '--inventory_only', '--halo_rings', '1', '2',
                 '--region_residues', '1', '--selection_seed', '43002')
    protein, = report['proteins']
    for entry in protein['seed_regions'].values():
        assert entry['feasible'] == 0
        assert len(entry['infeasible']) == protein['groups']
    assert 'observations' not in protein
    assert 'precision_state' not in report


@pytest.mark.parametrize('value, category', [(0.0, 'zero_reference'),
                                           (1e-7, 'near_zero_reference'),
                                           (float('nan'), 'nonfinite')])
def test_cli_distinguishes_zero_reference_from_nonfinite(tmp_path, checkpoint, value, category):
    saved = torch.load(checkpoint, weights_only=False)
    for tensor in saved['state_dict'].values():
        if tensor.is_floating_point():
            tensor.fill_(0 if category == 'near_zero_reference' else value)
    if category == 'near_zero_reference':
        saved['state_dict']['model.emb_to_latent.linear.bias'].fill_(value)
    torch.save(saved, checkpoint)
    report = run(tmp_path, checkpoint, '--halo_rings', '1', '--region_residues', str(L))
    strata = report['proteins'][0]['strata']
    assert all(row[category] == row['samples'] for row in strata)
    assert all(row['relative']['count'] == 0 for row in strata)
    if category == 'zero_reference':
        assert all(row['nonfinite'] == 0 and row['l2']['max'] == 0 for row in strata)
        assert report['summary']['1']['relative']['nonfinite'] == 0
        assert report['summary']['1']['relative']['undefined'] > 0
    json.dumps(report, allow_nan=False)


def test_cli_all_fixed_groups_infeasible_still_writes_a_report(tmp_path, checkpoint):
    report = run(tmp_path, checkpoint, '--halo_rings', '1', '2',
                 '--region_residues', '1', '--seed_groups', '0', '1')
    protein, = report['proteins']
    assert protein['coverage']['attempted'] == protein['coverage']['skipped'] == 2
    assert protein['coverage']['feasible'] == 0
    assert protein['regions'] == protein['observations'] == []


def test_capacity_mode_measures_regions_without_retaining_a_reference(tmp_path, checkpoint):
    report = run(tmp_path, checkpoint, '--halo_rings', '0', '1', '--capacity_only',
                 '--seed_groups', '0')
    protein, = report['proteins']
    assert protein['observations'] == []
    assert protein['measurements']['full'][0]['status'] == 'completed'
    assert len(protein['measurements']['regions']) == 4  # two t, two halos
    assert all(row['status'] == 'completed' for row in protein['measurements']['regions'])
