"""Offline full-protein partitions and the ghost drift report (#12 Ticket 2)."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from mdgen.full_length.partition import PartitionParams, build_spatial_partition, partitions_from_args
from mdgen.full_length.partition_store import ghost_drift, load_partition, save_partition

ROOT = Path(__file__).resolve().parents[1]
PARAMS = PartitionParams(cell_width=8.0, contact_radius=6.0, core_capacity=4, ghost_capacity=12)


def chain_ca(length=24, chains=2):
    ca = torch.zeros(length, 3)
    ca[:, 0] = 3.8 * torch.arange(length)
    ca[length // chains:, 1] = 4.0  # second chain runs alongside the first
    chain_id = torch.arange(length) // (length // chains)
    residue_index = torch.arange(length) % (length // chains)
    return ca, chain_id, residue_index


def build(ca, chain_id, residue_index, params=PARAMS):
    return build_spatial_partition(
        ca, mask=torch.ones(len(ca), dtype=torch.bool), chain_id=chain_id,
        residue_index=residue_index, **params.partition_kwargs())


def assert_same_partition(loaded, built):
    assert len(loaded.groups) == len(built.groups)
    for a, b in zip(loaded.groups, built.groups):
        assert a.cell == b.cell
        assert torch.equal(a.core_indices, b.core_indices)
        assert torch.equal(a.ghost_indices, b.ghost_indices)
        assert a.diagnostics == b.diagnostics
    assert torch.equal(loaded.owner_group, built.owner_group)
    assert torch.equal(loaded.residue_mask, built.residue_mask)
    assert loaded.diagnostics() == built.diagnostics()


def test_saved_partition_loads_back_identically(tmp_path):
    ca, chain_id, residue_index = chain_ca()
    built = build(ca, chain_id, residue_index)
    save_partition(tmp_path / 'p.npz', built, PARAMS, chain_id, residue_index)
    assert_same_partition(load_partition(tmp_path / 'p.npz', PARAMS, chain_id, residue_index), built)


def test_loading_with_different_numbering_raises(tmp_path):
    ca, chain_id, residue_index = chain_ca()
    save_partition(tmp_path / 'p.npz', build(ca, chain_id, residue_index), PARAMS,
                   chain_id, residue_index)
    with pytest.raises(ValueError, match='chain numbering'):  # e.g. prep_sims rerun found new chains
        load_partition(tmp_path / 'p.npz', PARAMS, torch.zeros_like(chain_id), residue_index)
    with pytest.raises(ValueError, match='residues'):
        load_partition(tmp_path / 'p.npz', PARAMS, chain_id[:-1], residue_index[:-1])


def test_generation_partitions_use_the_same_params():
    from types import SimpleNamespace
    args = SimpleNamespace(spatial_cell_width=8.0, spatial_contact_radius=6.0,
                           core_capacity=4, ghost_capacity=12)
    assert PartitionParams.from_args(args) == PARAMS
    ca, chain_id, residue_index = chain_ca()
    online, = partitions_from_args(args, ca[None], torch.ones(1, len(ca)), chain_id[None],
                                   residue_index[None])
    assert_same_partition(online, build(ca, chain_id, residue_index))


@pytest.mark.parametrize('field', ['cell_width', 'contact_radius', 'core_capacity',
                                   'ghost_capacity', 'covalent_span'])
def test_loading_with_different_params_raises(tmp_path, field):
    ca, chain_id, residue_index = chain_ca()
    save_partition(tmp_path / 'p.npz', build(ca, chain_id, residue_index), PARAMS,
                   chain_id, residue_index)
    changed = PartitionParams(**{**PARAMS.__dict__, field: getattr(PARAMS, field) + 1})
    with pytest.raises(ValueError, match=field):
        load_partition(tmp_path / 'p.npz', changed, chain_id, residue_index)


def test_drift_is_zero_on_frame_zero_and_positive_after_a_residue_moves():
    ca, chain_id, residue_index = chain_ca()
    partition = build(ca, chain_id, residue_index)
    moved = ca.clone()
    moved[20] = ca[2] + torch.tensor([0.0, 0.0, 1.0])  # far residue lands next to residue 2
    report = ghost_drift(partition, torch.stack([ca, moved]), PARAMS.contact_radius)

    assert report['frames'][0]['missing'] == 0 and report['frames'][0]['fraction'] == 0.0
    assert report['frames'][1]['missing'] > 0 and report['frames'][1]['fraction'] > 0.0
    assert report['frame0_spatial_truncated'] == 0
    assert report['max_fraction'] == report['frames'][1]['fraction']


def test_capacity_truncation_shows_up_on_frame_zero():
    ca, chain_id, residue_index = chain_ca()
    # 12 Å reaches 6 spatial neighbours per full core; capacity 4 keeps only the 4 covalent ones
    tight = PartitionParams(cell_width=8.0, contact_radius=12.0, core_capacity=4, ghost_capacity=4)
    partition = build(ca, chain_id, residue_index, tight)
    report = ghost_drift(partition, ca[None], tight.contact_radius)
    truncated = sum(group.diagnostics.spatial_truncated_count for group in partition.groups)
    assert truncated > 0
    assert report['frame0_spatial_truncated'] == truncated == report['frames'][0]['missing']


def write_replicate(data_dir, stem, length=24, frames=5):
    ca, chain_id, residue_index = chain_ca(length)
    atoms = np.zeros((frames, length, 14, 3), dtype=np.float32)
    atoms[:, :, 1] = ca.numpy()  # atom14 slot 1 is CA
    atoms[-1, length - 1] = atoms[0, 2] + [0.0, 0.0, 1.0]  # last frame: last residue drifts in
    np.save(data_dir / f'{stem}.npy', atoms)
    np.savez(data_dir / f'{stem}_chain.npz', chain_id=chain_id.numpy(),
             residue_index=residue_index.numpy())
    return ca, chain_id, residue_index


def test_cli_builds_every_replicate_and_writes_drift_reports(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    ca, chain_id, residue_index = write_replicate(data, 'prot_R1_i40')
    write_replicate(data, 'prot_R2_i40')
    split = tmp_path / 'split.csv'
    split.write_text('name,seqres\nprot,' + 'G' * 24 + '\n')
    out = tmp_path / 'partitions'
    subprocess.run([sys.executable, '-m', 'mdgen.full_length.partition_store',
                    '--split', str(split), '--data_dir', str(data), '--out_dir', str(out),
                    '--suffix', '_i40', '--atlas', '--replicates', '1', '2',
                    '--spatial_cell_width', '8', '--spatial_contact_radius', '6',
                    '--core_capacity', '4', '--ghost_capacity', '12', '--drift_stride', '2'],
                   cwd=ROOT, check=True, env={**os.environ, 'PYTHONPATH': str(ROOT)},
                   capture_output=True)

    assert sorted(path.name for path in out.iterdir()) == [
        'prot_R1_i40_drift.json', 'prot_R1_i40_partition.npz',
        'prot_R2_i40_drift.json', 'prot_R2_i40_partition.npz', 'summary.json']
    assert_same_partition(load_partition(out / 'prot_R1_i40_partition.npz', PARAMS,
                                         chain_id, residue_index),
                          build(ca, chain_id, residue_index))
    drift = json.loads((out / 'prot_R1_i40_drift.json').read_text())
    assert [frame['frame'] for frame in drift['frames']] == [0, 2, 4]
    assert drift['frames'][0]['missing'] == 0 and drift['frames'][-1]['missing'] > 0
    assert drift['params'] == PARAMS.__dict__
    summary = json.loads((out / 'summary.json').read_text())
    assert [entry['stem'] for entry in summary['replicates']] == ['prot_R1_i40', 'prot_R2_i40']


def run_cli(tmp_path, *extra, seqres_length=24):
    data = tmp_path / 'data'
    data.mkdir(exist_ok=True)
    write_replicate(data, 'prot')
    split = tmp_path / 'split.csv'
    split.write_text('name,seqres\nprot,' + 'G' * seqres_length + '\n')
    return subprocess.run([sys.executable, '-m', 'mdgen.full_length.partition_store',
                           '--split', str(split), '--data_dir', str(data),
                           '--out_dir', str(tmp_path / 'out'), *extra],
                          cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT)},
                          capture_output=True, text=True)


@pytest.mark.parametrize('stride', ['0', '-1'])
def test_cli_rejects_a_non_positive_drift_stride(tmp_path, stride):
    result = run_cli(tmp_path, '--drift_stride', stride)
    assert result.returncode != 0 and 'drift_stride' in result.stderr
    assert not (tmp_path / 'out').exists()


def test_cli_rejects_an_npy_that_does_not_match_seqres(tmp_path):
    result = run_cli(tmp_path, seqres_length=23)
    assert result.returncode != 0 and 'seqres' in result.stderr
