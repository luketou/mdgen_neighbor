"""Halo approximation error experiment (#12 Ticket 5), on a tiny random model."""
import json

import numpy as np
import pytest
import torch

from mdgen.full_length.group_region import sample_region_groups, with_halo
from mdgen.full_length.halo_error import boundary_rings, main
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
def checkpoint(monkeypatch, tmp_path):
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
    path = tmp_path / 'tiny.ckpt'
    torch.save({'hyper_parameters': {'args': args}, 'state_dict': wrapper.state_dict()}, path)
    return path


def run(tmp_path, checkpoint, *extra):
    split = write_protein(tmp_path)
    output = tmp_path / 'report.json'
    main(['--sim_ckpt', str(checkpoint), '--data_dir', str(tmp_path), '--split', str(split),
          '--replicate', '1', '--output', str(output), '--region_residues', '80',
          '--regions', '2', '--times', '0.3', '0.7', *LOCAL[1:], *extra])
    return json.loads(output.read_text())


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


def test_region_budget_counts_the_deepest_halo(tmp_path, checkpoint):
    with pytest.raises(ValueError, match='region_residues'):
        run(tmp_path, checkpoint, '--halo_rings', '1', '200')


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
