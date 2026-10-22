"""Chain numbering from topology and configurable replicates (#12 Ticket 1)."""
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import mdtraj
import numpy as np
import pytest
import torch

from mdgen.dataset import MDGenDataset
from mdgen.full_length.partition import build_spatial_partition
from mdgen.topology import chain_numbering

ROOT = Path(__file__).resolve().parents[1]
T = 3


def topology(chains):
    top = mdtraj.Topology()
    for numbers in chains:
        chain = top.add_chain()
        for number in numbers:
            residue = top.add_residue('GLY', chain, resSeq=number)
            for atom, element in (('N', 'N'), ('CA', 'C'), ('C', 'C'), ('O', 'O'), ('H', 'H')):
                top.add_atom(atom, mdtraj.element.get_by_symbol(element), residue)
    return top


def test_chain_numbering_follows_topology_chains_and_gaps():
    chain_id, residue_index = chain_numbering(topology([(1, 2, 4), (1, 2)]))
    assert chain_id.tolist() == [0, 0, 0, 1, 1]
    assert residue_index.tolist() == [1, 2, 4, 1, 2]
    assert chain_id.dtype == residue_index.dtype == np.int64


def test_consecutive_numbers_across_chains_are_not_covalent_neighbours():
    # Chain A ends at 5 and chain B starts at 6: numbering alone would join them.
    chain_id, residue_index = chain_numbering(topology([range(1, 6), range(6, 11)]))
    ca = torch.zeros(10, 3)
    ca[:, 0] = torch.arange(10) * 3.8
    ca[5:, 1] = 100.0  # chain B far away in space
    partition = build_spatial_partition(
        ca, mask=torch.ones(10, dtype=torch.bool), chain_id=torch.from_numpy(chain_id),
        residue_index=torch.from_numpy(residue_index), cell_width=200.0,
        contact_radius=4.0, core_capacity=1, ghost_capacity=4)
    last_of_a = partition.groups[int(partition.owner_group[4])]
    assert 5 not in last_of_a.ghost_indices.tolist()
    assert {2, 3} <= set(last_of_a.ghost_indices.tolist())


def write_atlas_sim(sim_dir, name, replicates, chains=((1, 2, 3), (1, 2, 3))):
    top = topology(chains)
    rng = np.random.default_rng(0)
    (sim_dir / name).mkdir(parents=True)
    for replicate in replicates:
        traj = mdtraj.Trajectory(rng.normal(size=(T, top.n_atoms, 3)).astype(np.float32), top)
        traj.save_xtc(str(sim_dir / name / f'{name}_prod_R{replicate}_fit.xtc'))
    traj.save_pdb(str(sim_dir / name / f'{name}.pdb'))


def test_prep_sims_writes_chain_files_for_requested_replicates(tmp_path):
    write_atlas_sim(tmp_path / 'sims', 'prot', (1, 2, 5))
    split = tmp_path / 'split.csv'
    split.write_text('name,seqres\nprot,GGGGGG\n')
    subprocess.run([sys.executable, 'scripts/prep_sims.py', '--split', str(split),
                    '--sim_dir', str(tmp_path / 'sims'), '--outdir', str(tmp_path / 'out'),
                    '--atlas', '--replicates', '2', '5'], cwd=ROOT, check=True,
                   env={**os.environ, 'PYTHONPATH': str(ROOT)}, capture_output=True)

    written = sorted(path.name for path in (tmp_path / 'out').iterdir())
    assert written == ['prot_R2.npy', 'prot_R2_chain.npz', 'prot_R5.npy', 'prot_R5_chain.npz']
    chain = np.load(tmp_path / 'out' / 'prot_R5_chain.npz')
    assert chain['chain_id'].tolist() == [0, 0, 0, 1, 1, 1]
    assert chain['residue_index'].tolist() == [1, 2, 3, 1, 2, 3]
    assert np.load(tmp_path / 'out' / 'prot_R5.npy').shape == (T, 6, 14, 3)


def dataset_args(tmp_path, **kwargs):
    return SimpleNamespace(**{
        **dict(data_dir=str(tmp_path), suffix='', frame_interval=None, num_frames=T,
               overfit=False, overfit_peptide=None, overfit_frame=False, copy_frames=False,
               no_frames=False, atlas=False, crop=256, training_region=False,
               replicates=[1, 2, 3]),
        **kwargs})


def write_npy(tmp_path, name, length):
    atoms = np.random.default_rng(0).normal(size=(T, length, 14, 3)).astype(np.float32)
    atoms[:, :, 1] = 0.0
    atoms[:, :, 1, 0] = 3.8 * np.arange(length)  # exact CA: residue i sits at x = 3.8 i
    np.save(tmp_path / f'{name}.npy', atoms)


def test_dataset_reads_chain_file_and_crops_it_with_the_residues(tmp_path):
    (tmp_path / 'split.csv').write_text('name,seqres\nprot,' + 'G' * 10 + '\n')
    write_npy(tmp_path, 'prot', 10)
    chain_id, residue_index = np.repeat([0, 1], 5), np.tile(np.arange(1, 6), 2)
    np.savez(tmp_path / 'prot_chain.npz', chain_id=chain_id, residue_index=residue_index)

    item = MDGenDataset(dataset_args(tmp_path), tmp_path / 'split.csv')[0]
    assert item['chain_id'].tolist() == chain_id.tolist()
    assert item['residue_index'].tolist() == residue_index.tolist()

    for replicate in (1, 2, 3):
        np.save(tmp_path / f'prot_R{replicate}.npy', np.load(tmp_path / 'prot.npy'))
        np.savez(tmp_path / f'prot_R{replicate}_chain.npz',
                 chain_id=chain_id, residue_index=residue_index)
    np.random.seed(0)
    cropped = MDGenDataset(dataset_args(tmp_path, atlas=True, crop=4), tmp_path / 'split.csv')[0]
    start = round(float(cropped['trans'][0, 0, 0]) / 3.8)
    assert cropped['chain_id'].tolist() == chain_id[start:start + 4].tolist()
    assert cropped['residue_index'].tolist() == residue_index[start:start + 4].tolist()

    padded = MDGenDataset(dataset_args(tmp_path, atlas=True, crop=12), tmp_path / 'split.csv')[0]
    assert padded['chain_id'].tolist() == chain_id.tolist() + [0, 0]
    assert padded['mask'].tolist() == [1] * 10 + [0, 0]


def test_dataset_without_chain_file_keeps_its_output(tmp_path):
    (tmp_path / 'split.csv').write_text('name,seqres\nprot,' + 'G' * 10 + '\n')
    write_npy(tmp_path, 'prot', 10)
    item = MDGenDataset(dataset_args(tmp_path), tmp_path / 'split.csv')[0]
    assert 'chain_id' not in item and 'residue_index' not in item


def test_atlas_replicates_come_from_the_configured_list(tmp_path):
    (tmp_path / 'split.csv').write_text('name,seqres\nprot,' + 'G' * 10 + '\n')
    write_npy(tmp_path, 'prot_R5', 10)
    dataset = MDGenDataset(dataset_args(tmp_path, atlas=True, replicates=[5]), tmp_path / 'split.csv')
    assert {dataset[0]['name'] for _ in range(5)} == {'prot_R5'}


def test_default_replicates_reproduce_the_previous_draws(tmp_path):
    (tmp_path / 'split.csv').write_text('name,seqres\nprot,' + 'G' * 10 + '\n')
    for replicate in (1, 2, 3):
        write_npy(tmp_path, f'prot_R{replicate}', 10)
    dataset = MDGenDataset(dataset_args(tmp_path, atlas=True), tmp_path / 'split.csv')
    np.random.seed(7)
    names = [dataset[0]['name'] for _ in range(20)]
    np.random.seed(7)  # the replicate is the only draw here (L < crop, one frame window)
    assert names == [f'prot_R{np.random.randint(1, 4)}' for _ in range(20)]
    assert len(set(names)) > 1


@pytest.mark.parametrize('argv, expected', [((), [1, 2, 3]), (('--replicates', '1', '4'), [1, 4])])
def test_replicates_cli(monkeypatch, argv, expected):
    from mdgen.parsing import parse_train_args
    monkeypatch.setattr('sys.argv', ['train.py', '--train_split', 'x', '--val_split', 'x',
                                     '--data_dir', 'x', *argv])
    assert parse_train_args().replicates == expected
