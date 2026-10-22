"""MDGenDataset Training region mode (#15 Ticket 2)."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mdgen.dataset import MDGenDataset

T = 3
PER_RESIDUE = ('torsions', 'trans', 'rots')  # [T, L, ...]
PER_POSITION = ('seqres', 'mask', 'torsion_mask')  # [L, ...]


def write_protein(tmp_path, name, length, extra=None):
    rng = np.random.default_rng(0)
    ca = np.zeros((length, 3), dtype=np.float32)
    ca[:, 0] = 3.8 * np.arange(length)  # straight chain: distance = index gap × 3.8 Å
    atoms = ca[None, :, None] + rng.normal(size=(T, length, 14, 3))
    atoms[:, :, 1] = ca  # atom14 slot 1 is CA
    np.save(tmp_path / f'{name}.npy', atoms.astype(np.float32))
    row = {'name': name, 'seqres': 'G' * length, **(extra or {})}
    split = tmp_path / f'{name}.csv'
    split.write_text(','.join(row) + '\n' + ','.join(
        f'"{value}"' if ',' in str(value) else str(value) for value in row.values()) + '\n')
    return split


def args(tmp_path, **kwargs):
    return SimpleNamespace(**{
        **dict(data_dir=str(tmp_path), suffix='', frame_interval=None, num_frames=T,
               overfit=False, overfit_peptide=None, overfit_frame=False, copy_frames=False,
               no_frames=False, atlas=False, crop=256, training_region=False,
               region_core=8, region_ghost=8, spatial_contact_radius=8.0),
        **kwargs})


def test_default_mode_output_is_unchanged(tmp_path):
    split = write_protein(tmp_path, 'prot', 40)
    item = MDGenDataset(args(tmp_path), split)[0]
    assert set(item) == {'name', 'frame_start', 'torsions', 'torsion_mask', 'trans',
                         'rots', 'seqres', 'mask', 'chain_id', 'residue_index'}
    assert item['trans'].shape == (T, 40, 3)
    # Without a chain file: the single-chain positional numbering the wrapper assumes anyway.
    assert item['chain_id'].tolist() == [0] * 40 and item['residue_index'].tolist() == list(range(40))

    for replica in (1, 2, 3):  # --atlas reads <name>_R{1,2,3}
        np.save(tmp_path / f'prot_R{replica}.npy', np.load(tmp_path / 'prot.npy'))
    cropped = MDGenDataset(args(tmp_path, atlas=True, crop=16), split)[0]
    assert 'core_mask' not in cropped and cropped['trans'].shape == (T, 16, 3)


def test_region_item_gathers_core_and_ghosts_in_residue_order(tmp_path):
    split = write_protein(tmp_path, 'prot', 40)
    full = MDGenDataset(args(tmp_path), split)[0]
    torch.manual_seed(3)
    item = MDGenDataset(args(tmp_path, training_region=True), split)[0]

    valid = item['mask'] > 0
    residues = item['residue_index'][valid]
    core = residues[item['core_mask'][valid] > 0]
    n = len(residues)
    assert np.all(np.diff(residues) > 0)  # one sequence-ordered stretch, not core then ghosts
    assert len(core) == 8 and np.all(np.diff(core) == 1)
    # 3.8 Å spacing and an 8 Å radius: two ghosts on each side, clipped at the chain ends
    expected_ghosts = {core[0] - 2, core[0] - 1, core[-1] + 1, core[-1] + 2} & set(range(40))
    assert set(residues) - set(core) == expected_ghosts
    assert np.all(valid[:n]) and not valid[n:].any() and not item['core_mask'][n:].any()
    for key in PER_RESIDUE:
        assert item[key].shape[:2] == (T, 16)
        torch.testing.assert_close(item[key][:, :n], full[key][:, residues])
    for key in PER_POSITION:
        assert len(item[key]) == 16
        np.testing.assert_array_equal(np.asarray(item[key][:n]),
                                      np.asarray(full[key])[residues])


def test_split_csv_chain_and_residue_numbering_are_used(tmp_path):
    chain = [0] * 20 + [1] * 20
    numbering = list(range(20)) * 2
    split = write_protein(tmp_path, 'prot', 40, {
        'chain_id': json.dumps(chain, separators=(',', ':')),
        'residue_index': json.dumps(numbering, separators=(',', ':'))})
    item = MDGenDataset(args(tmp_path, training_region=True, region_core=40), split)[0]

    valid = item['mask'] > 0
    np.testing.assert_array_equal(item['chain_id'][valid], chain)
    np.testing.assert_array_equal(item['residue_index'][valid], numbering)


def test_short_protein_is_all_core(tmp_path):
    split = write_protein(tmp_path, 'short', 6)
    item = MDGenDataset(args(tmp_path, training_region=True), split)[0]
    assert item['trans'].shape == (T, 16, 3)
    assert item['core_mask'].sum() == item['mask'].sum() == 6


def test_region_seed_fixes_validation_regions(tmp_path):
    split = write_protein(tmp_path, 'prot', 40)
    dataset = MDGenDataset(args(tmp_path, training_region=True), split, region_seed=0)
    centers = set()
    for _ in range(3):
        torch.manual_seed(len(centers))  # the global RNG must not matter
        centers.add(tuple(dataset[0]['residue_index'][dataset[0]['core_mask'] > 0]))
    assert len(centers) == 1


def test_region_seed_also_fixes_frame_window(tmp_path):
    split = write_protein(tmp_path, 'prot', 40)
    atoms = np.load(tmp_path / 'prot.npy')
    rng = np.random.default_rng(1)
    long = np.concatenate([atoms + rng.normal(scale=2.0, size=atoms.shape) for _ in range(4)])
    np.save(tmp_path / 'prot.npy', long.astype(np.float32))  # 12 frames, windows of 3
    dataset = MDGenDataset(args(tmp_path, training_region=True), split, region_seed=0)
    items = []
    for seed in range(4):
        np.random.seed(seed)
        torch.manual_seed(seed)
        items.append(dataset[0])
    assert len({item['frame_start'] for item in items}) == 1
    assert all(torch.equal(item['trans'], items[0]['trans']) for item in items)


@pytest.mark.parametrize('extra', [('--hyena',), ('--design',), ('--abs_pos_emb',)])
def test_training_region_rejects_incompatible_modes(monkeypatch, extra):
    from mdgen.parsing import parse_train_args
    monkeypatch.setattr('sys.argv', ['train.py', '--train_split', 'x', '--val_split', 'x',
                                     '--data_dir', 'x', '--training_region', *extra])
    with pytest.raises(SystemExit):
        parse_train_args()


def test_abs_pos_emb_is_allowed_when_region_width_matches_crop(monkeypatch):
    from mdgen.parsing import parse_train_args
    monkeypatch.setattr('sys.argv', ['train.py', '--train_split', 'x', '--val_split', 'x',
                                     '--data_dir', 'x', '--training_region', '--abs_pos_emb',
                                     '--region_core', '128', '--region_ghost', '128'])
    assert parse_train_args().crop == 256


@pytest.mark.parametrize('extra, ok', [((), True), (('--no_frames',), False)])
def test_training_region_cli(monkeypatch, extra, ok):
    from mdgen.parsing import parse_train_args
    monkeypatch.setattr('sys.argv', ['train.py', '--train_split', 'x', '--val_split', 'x',
                                     '--data_dir', 'x', '--training_region', *extra])
    if not ok:
        with pytest.raises(SystemExit):
            parse_train_args()
        return
    cli = parse_train_args()
    assert cli.training_region and (cli.region_core, cli.region_ghost) == (256, 256)
