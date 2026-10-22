"""Training region sampler: one random spatial core plus its ghost context (#15 Ticket 1)."""
import pytest
import torch

from mdgen.full_length.region import sample_training_region


def line(count, spacing=1.0):
    return torch.stack([torch.tensor([i * spacing, 0.0, 0.0]) for i in range(count)])


def sample(ca, *, mask=None, chain_id=None, residue_index=None, **kwargs):
    L = ca.shape[0]
    return sample_training_region(
        ca,
        mask=torch.ones(L, dtype=torch.bool) if mask is None else mask,
        chain_id=torch.zeros(L, dtype=torch.long) if chain_id is None else chain_id,
        residue_index=torch.arange(L) if residue_index is None else residue_index,
        **kwargs,
    )


def test_core_is_nearest_residues_and_ghosts_are_contact_shell():
    region = sample(line(20), center=10, core_count=5, ghost_capacity=10,
                    contact_radius=3.0)

    assert region.center == 10
    assert region.core_indices.tolist() == [8, 9, 10, 11, 12]
    assert region.ghost_indices.tolist() == [5, 6, 7, 13, 14, 15]
    details = region.diagnostics
    assert (details.core_count, details.ghost_count) == (5, 6)
    assert details.mandatory_neighbor_count == details.mandatory_retained_count == 4
    assert details.spatial_neighbor_count == details.spatial_retained_count == 6
    assert details.spatial_truncated_count == 0


def test_ghost_cap_keeps_covalent_neighbours_and_drops_farthest():
    ca = line(20)
    ca[15, 0] = 14.5  # residue 15 is nearer the core than residue 5
    region = sample(ca, center=10, core_count=5, ghost_capacity=5, contact_radius=3.0)

    assert region.ghost_indices.tolist() == [6, 7, 13, 14, 15]
    assert region.diagnostics.mandatory_retained_count == 4
    assert region.diagnostics.spatial_truncated_count == 1

    covalent_only = sample(line(20), center=10, core_count=5, ghost_capacity=4,
                           contact_radius=3.0)
    assert covalent_only.ghost_indices.tolist() == [6, 7, 13, 14]
    assert covalent_only.diagnostics.spatial_truncated_count == 2


def test_covalent_neighbours_beyond_ghost_capacity_raise():
    with pytest.raises(ValueError, match='ghost_capacity'):
        sample(line(20), center=10, core_count=5, ghost_capacity=3, contact_radius=3.0)


def test_covalent_neighbours_do_not_cross_chains():
    ca = torch.cat([line(5), line(5) + torch.tensor([100.0, 0.0, 0.0])])
    region = sample(ca, chain_id=torch.tensor([0] * 5 + [1] * 5),
                    residue_index=torch.arange(10), center=4, core_count=2,
                    ghost_capacity=10, contact_radius=1.5)

    assert region.core_indices.tolist() == [3, 4]
    assert region.ghost_indices.tolist() == [1, 2]


def test_short_protein_is_all_core_and_masked_residues_are_never_selected():
    mask = torch.tensor([1, 1, 0, 1, 1, 1, 0], dtype=torch.bool)
    region = sample(line(7), mask=mask, core_count=8, ghost_capacity=4,
                    contact_radius=3.0, generator=torch.Generator().manual_seed(0))

    assert region.core_indices.tolist() == [0, 1, 3, 4, 5]
    assert region.ghost_indices.numel() == 0

    near_masked = sample(line(7), mask=mask, center=0, core_count=2, ghost_capacity=4,
                         contact_radius=3.0)
    assert 2 not in near_masked.ghost_indices.tolist()


def test_random_center_is_valid_and_reproducible_under_a_seed():
    mask = torch.ones(50, dtype=torch.bool)
    mask[::3] = False
    regions = [
        sample(line(50), mask=mask, core_count=4, ghost_capacity=8, contact_radius=2.0,
               generator=torch.Generator().manual_seed(seed))
        for seed in (7, 7, 8)
    ]

    assert all(mask[region.center] for region in regions)
    assert regions[0].center == regions[1].center
    assert torch.equal(regions[0].core_indices, regions[1].core_indices)
    assert torch.equal(regions[0].ghost_indices, regions[1].ghost_indices)


@pytest.mark.parametrize('kwargs, message', [
    (dict(core_count=0), 'core_count'),
    (dict(ghost_capacity=-1), 'ghost_capacity'),
    (dict(contact_radius=0.0), 'contact_radius'),
    (dict(center=2), 'center'),
])
def test_invalid_arguments_raise(kwargs, message):
    mask = torch.tensor([1, 1, 0, 1], dtype=torch.bool)
    options = dict(core_count=2, ghost_capacity=2, contact_radius=3.0, center=0) | kwargs
    with pytest.raises(ValueError, match=message):
        sample(line(4), mask=mask, **options)


def test_duplicate_residue_numbering_raises():
    with pytest.raises(ValueError, match='unique'):
        sample(line(4), residue_index=torch.tensor([0, 1, 1, 2]), center=0,
               core_count=2, ghost_capacity=2, contact_radius=3.0)
