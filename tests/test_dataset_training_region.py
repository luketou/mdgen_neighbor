"""MDGenDataset Training region mode on the frame-0 full-protein partition (#12 Ticket 4)."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mdgen.dataset import MDGenDataset, collate
from mdgen.full_length.group_region import region_subpartition, sample_region_groups
from mdgen.full_length.partition import PartitionParams, build_spatial_partition

T, L, WIDTH = 3, 200, 80


def write_protein(tmp_path, name='prot', length=L, frames=T, replicas=None):
    rng = np.random.default_rng(0)
    step = rng.normal(size=(length, 3)) + [0.5, 0, 0]  # loosely extended: small seed regions
    ca = (3.8 * step / np.linalg.norm(step, axis=-1, keepdims=True)).cumsum(0)
    atoms = ca[None, :, None] + rng.normal(size=(frames, length, 14, 3))
    atoms[:, :, 1] = ca + rng.normal(scale=0.3, size=(frames, length, 3))  # atom14 slot 1 is CA
    for stem in ([f'{name}_R{i}' for i in replicas] if replicas else [name]):
        np.save(tmp_path / f'{stem}.npy', atoms.astype(np.float32))
    split = tmp_path / f'{name}.csv'
    split.write_text(f'name,seqres\n{name},{"ACDEFGHIKLMNPQRSTVWY" * (length // 20)}\n')
    return split


def args(tmp_path, **kwargs):
    return SimpleNamespace(**{
        **dict(data_dir=str(tmp_path), suffix='', frame_interval=None, num_frames=T,
               overfit=False, overfit_peptide=None, overfit_frame=False, copy_frames=False,
               no_frames=False, atlas=False, crop=256, replicates=[1, 2, 3],
               training_region=True, region_residues=WIDTH, halo_rings=1, partition_dir=None,
               spatial_cell_width=8.0, spatial_contact_radius=8.0, core_capacity=8,
               ghost_capacity=32),
        **kwargs})


def positions(item):
    """Full-protein index of each region position (padding reads 0)."""
    return item['partition'].position_index.numpy()


def full_partition(tmp_path, name='prot'):
    ca = torch.from_numpy(np.load(tmp_path / f'{name}.npy')[0, :, 1])
    return ca, build_spatial_partition(
        ca, mask=torch.ones(L, dtype=torch.bool), chain_id=torch.zeros(L, dtype=torch.long),
        residue_index=torch.arange(L), **PartitionParams.from_args(args(tmp_path)).partition_kwargs())


def test_region_fields_match_the_full_protein_at_region_residues(tmp_path):
    split = write_protein(tmp_path)
    full = MDGenDataset(args(tmp_path, training_region=False), split)[0]
    item = MDGenDataset(args(tmp_path), split, region_seed=0)[0]

    n = item['region']['residues']
    residues = positions(item)[:n]
    assert np.all(np.diff(residues) > 0)
    # some residue's predecessor is outside the region, so phi / pre-omega need the extra read
    assert not np.isin(residues[residues > 0] - 1, residues).all()
    for key in ('torsions', 'trans', 'rots'):
        assert item[key].shape[:2] == (T, WIDTH)
        torch.testing.assert_close(item[key][:, :n], full[key][:, residues])
    for key in ('torsion_mask', 'seqres', 'chain_id', 'residue_index'):
        np.testing.assert_array_equal(np.asarray(item[key][:n]), np.asarray(full[key])[residues])
    assert item['mask'][:n].all() and not item['mask'][n:].any()
    assert not item['loss_core_mask'][n:].any()
    assert torch.equal(item['rots'][:, n:], torch.eye(3).expand(T, WIDTH - n, 3, 3))


def test_region_is_cut_from_the_frame0_partition_and_padded(tmp_path):
    split = write_protein(tmp_path)
    item = MDGenDataset(args(tmp_path), split, region_seed=0)[0]
    ca, partition = full_partition(tmp_path)
    selection = sample_region_groups(partition, ca, region_residues=WIDTH, halo_rings=1,
                                     generator=torch.Generator().manual_seed(0))
    expected = region_subpartition(partition, selection)

    n = len(expected.residues)
    sub = item['partition']
    assert n == item['region']['residues'] and sub.residue_mask.numel() == WIDTH
    assert torch.equal(sub.position_index[:n], expected.residues)
    assert torch.equal(sub.owner_group[:n], expected.partition.owner_group)
    assert (sub.owner_group[n:] == -1).all() and not sub.residue_mask[n:].any()
    assert sub.full_length == L and sub.validate_ownership().valid
    for got, want in zip(sub.groups, expected.partition.groups, strict=True):
        assert torch.equal(got.core_indices, want.core_indices)
        assert torch.equal(got.ghost_indices, want.ghost_indices)
    np.testing.assert_array_equal(item['loss_core_mask'][:n], expected.loss_core_mask.numpy())
    assert item['region']['halo_groups'] == len(selection.halo_groups) > 0


def test_stored_partition_is_used_and_checked(tmp_path):
    from mdgen.full_length.partition_store import main as store
    split = write_protein(tmp_path)
    store(['--split', str(split), '--data_dir', str(tmp_path), '--out_dir', str(tmp_path / 'p'),
           '--spatial_cell_width', '8', '--spatial_contact_radius', '8',
           '--core_capacity', '8', '--ghost_capacity', '32'])
    built = MDGenDataset(args(tmp_path), split, region_seed=0)[0]
    stored = MDGenDataset(args(tmp_path, partition_dir=str(tmp_path / 'p')), split,
                          region_seed=0)[0]
    np.testing.assert_array_equal(positions(stored), positions(built))

    stale = MDGenDataset(args(tmp_path, partition_dir=str(tmp_path / 'p'), core_capacity=6),
                         split, region_seed=0)
    with pytest.raises(ValueError, match='core_capacity'):
        stale[0]


def test_validation_regions_are_fixed_across_epochs(tmp_path):
    split = write_protein(tmp_path, frames=4 * T)  # several frame windows to choose from
    dataset = MDGenDataset(args(tmp_path), split, repeat=2, region_seed=0)
    epochs = []
    for epoch in range(3):
        np.random.seed(epoch)  # the global RNGs must not matter
        torch.manual_seed(epoch)
        epochs.append([dataset[i] for i in range(len(dataset))])
    for first, later in zip(epochs[0], epochs[1]):
        assert first['frame_start'] == later['frame_start']
        np.testing.assert_array_equal(positions(first), positions(later))
        assert torch.equal(first['trans'], later['trans'])
    # repeated validation copies are different, still fixed, regions
    assert not np.array_equal(positions(epochs[0][0]), positions(epochs[0][1]))


def test_training_regions_vary(tmp_path):
    split = write_protein(tmp_path)
    dataset = MDGenDataset(args(tmp_path), split)
    torch.manual_seed(0)
    regions = {tuple(positions(dataset[0])) for _ in range(8)}
    assert len(regions) > 1


def test_validation_replicates_are_separate(tmp_path):
    split = write_protein(tmp_path, replicas=[1, 2, 3, 4, 5])
    val = MDGenDataset(args(tmp_path, atlas=True, replicates=[1, 2, 3, 4]), split,
                       region_seed=0, replicates=[5])
    train = MDGenDataset(args(tmp_path, atlas=True, replicates=[1, 2, 3, 4]), split)
    assert {val[0]['name'] for _ in range(4)} == {'prot_R5'}
    np.random.seed(0)
    assert {train[0]['name'] for _ in range(40)} == {f'prot_R{i}' for i in range(1, 5)}


def test_region_budget_is_checked_at_startup(tmp_path):
    from mdgen.full_length.group_region import largest_seed_region
    split = write_protein(tmp_path)
    _, partition = full_partition(tmp_path)
    needed = largest_seed_region(partition, 1)
    MDGenDataset(args(tmp_path, region_residues=needed), split).check_region_budget()
    with pytest.raises(ValueError, match='region_residues'):
        MDGenDataset(args(tmp_path, region_residues=needed - 1), split).check_region_budget()


def test_collate_keeps_partitions_and_summaries_as_lists(tmp_path):
    split = write_protein(tmp_path)
    dataset = MDGenDataset(args(tmp_path), split, repeat=2, region_seed=0)
    batch = collate([dataset[0], dataset[1]])
    assert batch['trans'].shape == (2, T, WIDTH, 3) and batch['loss_core_mask'].shape == (2, WIDTH)
    assert len(batch['partition']) == 2 and len(batch['region']) == 2
    assert batch['partition'][1] is not batch['partition'][0]
    # one shared storage per item: a DataLoader worker sends one tensor, not two per group
    storages = {index.untyped_storage().data_ptr() for g in batch['partition'][0].groups
                for index in (g.core_indices, g.ghost_indices)}
    assert len(storages) == 1


LOCAL = ['--local_spatial', '--spatial_cell_width', '8', '--spatial_contact_radius', '8',
         '--core_capacity', '8', '--ghost_capacity', '32']


def cli(monkeypatch, *extra):
    from mdgen.parsing import parse_train_args
    monkeypatch.setattr('sys.argv', ['train.py', '--train_split', 'x', '--val_split', 'x',
                                     '--data_dir', 'x', '--training_region', *extra])
    return parse_train_args()


def test_training_region_cli_defaults(monkeypatch):
    parsed = cli(monkeypatch, *LOCAL)
    assert (parsed.region_residues, parsed.halo_rings, parsed.partition_dir) == (1024, 1, None)
    assert parsed.val_replicates is None


@pytest.mark.parametrize('extra', [(), ('--hyena',), ('--design',), ('--abs_pos_emb',),
                                   ('--halo_rings', '-1')])
def test_training_region_rejects_incompatible_modes(monkeypatch, extra):
    local = () if extra == () else tuple(LOCAL)  # () checks that --local_spatial is required
    with pytest.raises(SystemExit):
        cli(monkeypatch, *local, *extra)


def tiny_wrapper(monkeypatch, tmp_path):
    from mdgen.wrapper import NewMDGenWrapper
    monkeypatch.setenv('MODEL_DIR', str(tmp_path / 'model'))
    parsed = cli(monkeypatch, *LOCAL, '--num_layers', '1', '--embed_dim', '32', '--mha_heads', '4',
                 '--ipa_heads', '2', '--ipa_head_dim', '4', '--ipa_qk', '2', '--ipa_v', '2',
                 '--num_frames', str(T), '--prepend_ipa', '--interleave_ipa', '--sim_condition',
                 '--data_dir', str(tmp_path), '--region_residues', str(WIDTH))
    delattr(parsed, 'manual_attention')
    torch.manual_seed(0)
    return parsed, NewMDGenWrapper(parsed)


def test_wrapper_uses_the_batch_partition_and_loss_core_mask(monkeypatch, tmp_path):
    split = write_protein(tmp_path)
    parsed, model = tiny_wrapper(monkeypatch, tmp_path)
    batch = collate([MDGenDataset(parsed, split, repeat=2, region_seed=0)[i] for i in range(2)])

    def no_repartition(*args, **kwargs):
        raise AssertionError('a Training region batch must not be re-partitioned')
    monkeypatch.setattr('mdgen.full_length.partition.partitions_from_args', no_repartition)
    prep = model.prep_batch(batch)
    assert prep['model_kwargs']['spatial_partitions'] is batch['partition']
    outside = batch['loss_core_mask'] == 0
    assert prep['loss_mask'].permute(0, 2, 1, 3)[outside].abs().sum() == 0
    assert prep['loss_mask'].permute(0, 2, 1, 3)[~outside].abs().sum() > 0


def test_one_training_step_on_a_region_batch(monkeypatch, tmp_path):
    import pytorch_lightning as pl
    split = write_protein(tmp_path)
    parsed, model = tiny_wrapper(monkeypatch, tmp_path)
    loader = torch.utils.data.DataLoader(MDGenDataset(parsed, split, repeat=2), batch_size=2,
                                         collate_fn=collate)
    before = {k: v.detach().clone() for k, v in model.model.named_parameters()}
    trainer = pl.Trainer(accelerator='cpu', max_steps=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, limit_val_batches=0)
    trainer.fit(model, loader)
    changed = [k for k, v in model.model.named_parameters() if not torch.equal(v, before[k])]
    assert changed
    assert all(v.isfinite().all() for v in model.model.parameters())


def test_split_csv_chain_columns_are_rejected(tmp_path):
    write_protein(tmp_path)
    split = tmp_path / 'chains.csv'
    split.write_text('name,seqres,chain_id\nprot,AC,"[0,1]"\n')
    with pytest.raises(ValueError, match='_chain.npz'):
        MDGenDataset(args(tmp_path), split)
    MDGenDataset(args(tmp_path, training_region=False), split)  # ignored without regions


def write_chains(tmp_path, name='prot'):
    """Two chains of 100 with a numbering gap after residue 39 of chain A."""
    chain_id = np.repeat([0, 1], L // 2)
    residue_index = np.concatenate([np.arange(1, 41), np.arange(45, 105), np.arange(1, 101)])
    np.savez(tmp_path / f'{name}_chain.npz', chain_id=chain_id, residue_index=residue_index)
    return [0, 40, L // 2]  # residues without a same-chain, consecutive predecessor


def test_pre_omega_and_phi_are_masked_across_chain_breaks(tmp_path):
    split = write_protein(tmp_path)
    plain = MDGenDataset(args(tmp_path, training_region=False), split)[0]
    breaks = write_chains(tmp_path)
    full = MDGenDataset(args(tmp_path, training_region=False), split)[0]
    expected = plain['torsion_mask'].clone()
    expected[breaks, :2] = 0
    assert plain['torsion_mask'][breaks[1:], :2].all()  # residue 0 is masked anyway; the rest are new
    torch.testing.assert_close(full['torsion_mask'], expected)
    # every region residue keeps the full-protein mask, breaks included
    torch.manual_seed(0)
    dataset = MDGenDataset(args(tmp_path), split)
    seen = set()
    for _ in range(20):
        item = dataset[0]
        residues = positions(item)[:item['region']['residues']]
        torch.testing.assert_close(item['torsion_mask'][:len(residues)], expected[residues])
        seen |= set(residues.tolist()) & set(breaks)
    assert seen  # some sampled region contains a chain break
