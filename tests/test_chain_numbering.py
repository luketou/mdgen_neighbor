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
                    '--atlas', '--replicates', '2', '5', '--suffix', '_i1'], cwd=ROOT, check=True,
                   env={**os.environ, 'PYTHONPATH': str(ROOT)}, capture_output=True)

    written = sorted(path.name for path in (tmp_path / 'out').iterdir())
    assert written == ['prot_R2_i1.npy', 'prot_R2_i1_chain.npz',
                       'prot_R5_i1.npy', 'prot_R5_i1_chain.npz']
    chain = np.load(tmp_path / 'out' / 'prot_R5_i1_chain.npz')
    assert chain['chain_id'].tolist() == [0, 0, 0, 1, 1, 1]
    assert chain['residue_index'].tolist() == [1, 2, 3, 1, 2, 3]
    assert np.load(tmp_path / 'out' / 'prot_R5_i1.npy').shape == (T, 6, 14, 3)


def test_prep_sims_writes_nothing_for_a_topology_it_rejects(tmp_path):
    (tmp_path / 'sims' / 'prot').mkdir(parents=True)
    pdb = tmp_path / 'sims' / 'prot' / 'prot.pdb'
    lines = []  # residue 2 has an insertion-code twin 2A, so resSeq repeats
    for k, (number, icode) in enumerate([(1, ' '), (2, ' '), (2, 'A'), (3, ' ')]):
        for j, (atom, element) in enumerate((('N', 'N'), ('CA', 'C'), ('C', 'C'), ('O', 'O'))):
            lines.append(f'ATOM  {4 * k + j + 1:5d}  {atom:<3s} GLY A{number:4d}{icode}   '
                         f'{3.8 * k + j:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00           {element}')
    pdb.write_text('\n'.join(lines) + '\nEND\n')
    mdtraj.load(str(pdb)).save_xtc(str(tmp_path / 'sims' / 'prot' / 'prot_prod_R1_fit.xtc'))
    split = tmp_path / 'split.csv'
    split.write_text('name,seqres\nprot,GGGG\n')
    result = subprocess.run([sys.executable, 'scripts/prep_sims.py', '--split', str(split),
                             '--sim_dir', str(tmp_path / 'sims'), '--outdir', str(tmp_path / 'out'),
                             '--atlas', '--replicates', '1'], cwd=ROOT,
                            env={**os.environ, 'PYTHONPATH': str(ROOT)}, capture_output=True, text=True)
    assert result.returncode != 0 and 'chain 0 residue 2' in result.stderr
    assert list((tmp_path / 'out').iterdir()) == []  # nothing half-written is left behind


def test_prep_sims_rerun_completes_an_output_missing_its_chain_file(tmp_path):
    sims = tmp_path / 'sims'
    (sims / 'prot').mkdir(parents=True)
    rng = np.random.default_rng(0)
    traj = mdtraj.Trajectory(rng.normal(size=(T, 15, 3)).astype(np.float32), topology([(1, 2, 3)]))
    traj.save_pdb(str(sims / 'prot' / 'prot.pdb'))
    traj.save_xtc(str(sims / 'prot' / 'prot.xtc'))
    (tmp_path / 'out').mkdir()
    np.save(tmp_path / 'out' / 'prot.npy', np.zeros(1))  # e.g. from a run killed mid-job
    split = tmp_path / 'split.csv'
    split.write_text('name,seqres\nprot,GGG\n')
    subprocess.run([sys.executable, 'scripts/prep_sims.py', '--split', str(split),
                    '--sim_dir', str(sims), '--outdir', str(tmp_path / 'out')], cwd=ROOT,
                   check=True, env={**os.environ, 'PYTHONPATH': str(ROOT)}, capture_output=True)
    assert np.load(tmp_path / 'out' / 'prot_chain.npz')['residue_index'].tolist() == [1, 2, 3]
    assert np.load(tmp_path / 'out' / 'prot.npy').shape == (T, 3, 14, 3)


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
    assert padded['residue_index'].tolist() == residue_index.tolist() + [0, 0]
    assert padded['mask'].tolist() == [1] * 10 + [0, 0]
    for replicate in (1, 2, 3):
        os.remove(tmp_path / f'prot_R{replicate}_chain.npz')
    default = MDGenDataset(dataset_args(tmp_path, atlas=True, crop=12), tmp_path / 'split.csv')[0]
    assert default['residue_index'].tolist() == list(range(10)) + [0, 0]  # pads match


def test_dataset_without_chain_file_uses_the_default_numbering(tmp_path):
    (tmp_path / 'split.csv').write_text('name,seqres\nprot,' + 'G' * 10 + '\n')
    write_npy(tmp_path, 'prot', 10)
    item = MDGenDataset(dataset_args(tmp_path), tmp_path / 'split.csv')[0]
    assert item['chain_id'].tolist() == [0] * 10  # same as the wrapper's own fallback
    assert item['residue_index'].tolist() == list(range(10))


def test_items_with_and_without_chain_files_collate_together(tmp_path):
    (tmp_path / 'split.csv').write_text('name,seqres\nwith,' + 'G' * 10 + '\nwithout,' + 'G' * 10 + '\n')
    write_npy(tmp_path, 'with', 10)
    write_npy(tmp_path, 'without', 10)
    np.savez(tmp_path / 'with_chain.npz', chain_id=np.repeat([0, 1], 5),
             residue_index=np.tile(np.arange(1, 6), 2))
    dataset = MDGenDataset(dataset_args(tmp_path), tmp_path / 'split.csv')
    for order in ((0, 1), (1, 0)):
        batch = torch.utils.data.default_collate([dataset[i] for i in order])
        with_row = order.index(0)
        assert batch['chain_id'][with_row].tolist() == [0] * 5 + [1] * 5
        assert batch['chain_id'][1 - with_row].tolist() == [0] * 10


def test_duplicate_residue_numbers_fail_at_preprocessing():
    with pytest.raises(ValueError, match='chain 0 residue 52'):
        chain_numbering(topology([(51, 52, 52, 53)]))  # e.g. 52 and 52A after dropping insertion codes


def load_sim_inference(monkeypatch, tmp_path, *argv):
    import importlib
    monkeypatch.setattr('sys.argv', ['sim_inference.py', '--sim_ckpt', 'x', '--data_dir',
                                     str(tmp_path), '--out_dir', str(tmp_path / 'out'), *argv])
    monkeypatch.delitem(sys.modules, 'sim_inference', raising=False)
    deterministic = torch.are_deterministic_algorithms_enabled()
    monkeypatch.setattr(torch.backends.cudnn, 'benchmark', torch.backends.cudnn.benchmark)
    try:
        module = importlib.import_module('sim_inference')  # sets global torch flags at import
    finally:
        torch.use_deterministic_algorithms(deterministic)
    monkeypatch.delitem(sys.modules, 'sim_inference')
    return module


@pytest.mark.parametrize('suffix', ['', '_R1', '_R1_i40'])
def test_inference_batch_uses_the_same_chain_file(monkeypatch, tmp_path, suffix):
    sim = load_sim_inference(monkeypatch, tmp_path, '--suffix', suffix)
    write_npy(tmp_path, f'prot{suffix}', 10)
    chain_id = np.repeat([0, 1], 5)  # the chain file sits next to the NPY: <stem>_chain.npz
    np.savez(tmp_path / f'prot{suffix}_chain.npz', chain_id=chain_id,
             residue_index=np.tile(np.arange(5), 2))
    item = sim.get_batch('prot', 'G' * 10, num_frames=T)
    assert item['chain_id'].tolist() == chain_id.tolist()

    seen = {}
    class Model:
        def inference(self, batch):
            seen.update(batch)
            raise StopIteration
    batch = tensor_tree(torch.utils.data.default_collate([item]))
    with pytest.raises(StopIteration):
        sim.rollout(Model(), batch)
    assert seen['chain_id'][0].tolist() == chain_id.tolist()
    assert 'residue_index' in seen


def tensor_tree(batch):
    return {key: torch.as_tensor(value) for key, value in batch.items()}


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
