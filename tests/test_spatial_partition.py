import json

import pytest
import torch

from mdgen.full_length.partition import SpatialPartition, build_spatial_partition


def test_partition_preserves_unique_ownership_and_covalent_boundary_context():
    ca = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [17.0, 0.0, 0.0],
            [18.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
        ]
    )
    partition = build_spatial_partition(
        ca,
        mask=torch.ones(5, dtype=torch.bool),
        chain_id=torch.tensor([0, 0, 0, 0, 1]),
        residue_index=torch.tensor([0, 1, 2, 3, 0]),
        cell_width=16.0,
        contact_radius=4.0,
        core_capacity=2,
        ghost_capacity=3,
    )

    assert isinstance(partition, SpatialPartition)
    assert partition.validate_ownership().duplicate_count == 0
    assert partition.validate_ownership().missing_count == 0
    assert max(group.core_indices.numel() for group in partition.groups) <= 2
    assert max(group.ghost_indices.numel() for group in partition.groups) <= 3

    owner_of_one = partition.owner_group[1].item()
    owner_group = partition.groups[owner_of_one]
    assert 2 in owner_group.ghost_indices.tolist()
    assert 4 in owner_group.ghost_indices.tolist()


def test_gather_scatter_uses_only_core_outputs_and_restores_global_order():
    ca = torch.stack([torch.tensor([float(i * 9), 0.0, 0.0]) for i in range(6)])
    partition = build_spatial_partition(
        ca,
        mask=torch.ones(6, dtype=torch.bool),
        chain_id=torch.zeros(6, dtype=torch.long),
        residue_index=torch.arange(6),
        cell_width=16.0,
        contact_radius=12.0,
        core_capacity=2,
        ghost_capacity=4,
    )
    states = torch.arange(18, dtype=torch.float32).reshape(6, 3)

    gathered = partition.gather(states)
    local_outputs = gathered.values.clone()
    local_outputs[gathered.ghost_mask] = -10_000.0
    restored = partition.scatter_core(local_outputs, gathered)

    torch.testing.assert_close(restored, states)
    assert partition.ghost_update_violations(local_outputs, gathered) == 0


def test_mandatory_neighbor_overflow_splits_cores_without_widening_capacity():
    # Two dimers cross the cell boundary. Each initial two-residue core needs
    # two mandatory ghosts; singleton cores can retain their partner in one.
    ca = torch.tensor([[0., 0., 0.], [20., 0., 0.], [1., 0., 0.], [21., 0., 0.]])
    options = dict(
        mask=torch.ones(4, dtype=torch.bool),
        chain_id=torch.tensor([0, 0, 1, 1]),
        residue_index=torch.tensor([0, 1, 0, 1]),
        cell_width=16., contact_radius=4., core_capacity=2,
        ghost_capacity=1, covalent_span=1,
    )
    partition = build_spatial_partition(ca, **options)
    repeated = build_spatial_partition(ca, **options)

    assert partition.ghost_capacity == 1
    assert partition.validate_ownership().valid
    assert [group.core_indices.tolist() for group in partition.groups] == [[0], [2], [1], [3]]
    for group, again in zip(partition.groups, repeated.groups):
        core_index = group.core_indices.item()
        assert group.ghost_indices.tolist() == [core_index ^ 1]
        torch.testing.assert_close(group.indices, again.indices)
        assert group.diagnostics.mandatory_neighbor_count == 1
        assert group.diagnostics.mandatory_retained_count == 1

    states = torch.arange(4, dtype=torch.float32).unsqueeze(-1)
    gathered = partition.gather(states)
    outputs = gathered.values.clone()
    outputs[gathered.ghost_mask] = 999.
    torch.testing.assert_close(partition.scatter_core(outputs, gathered), states)
    assert partition.ghost_update_violations(outputs, gathered) == 0


def test_mandatory_neighbors_that_cannot_fit_singleton_raise():
    ca = torch.tensor([[0., 0., 0.], [20., 0., 0.], [40., 0., 0.]])
    with pytest.raises(ValueError, match="ghost_capacity=1.*2 mandatory neighbors.*Increase ghost_capacity"):
        build_spatial_partition(
            ca, mask=torch.ones(3, dtype=torch.bool),
            chain_id=torch.zeros(3, dtype=torch.long), residue_index=torch.arange(3),
            core_capacity=1, ghost_capacity=1, covalent_span=1,
        )


def test_partition_diagnostics_distinguish_mandatory_and_spatial_retention():
    # The only mandatory neighbor of residue 0 is far away. Its one ghost slot
    # must go there, even though two other-chain spatial neighbors are nearby.
    ca = torch.tensor([[0., 0., 0.], [40., 0., 0.], [1., 0., 0.], [2., 0., 0.], [0., 0., 0.]])
    partition = build_spatial_partition(
        ca, mask=torch.tensor([True, True, True, True, False]),
        chain_id=torch.tensor([0, 0, 1, 2, 0]),
        residue_index=torch.tensor([0, 1, 0, 0, 2]),
        contact_radius=3., core_capacity=1, ghost_capacity=1, covalent_span=1,
    )
    report = partition.diagnostics()
    json.dumps(report)  # No tensors in the public report.
    assert report["ownership_valid"]
    assert report["owner_group_matches"]
    assert report["missing_count"] == report["duplicate_count"] == 0
    assert report["valid_residue_count"] == report["group_count"] == 4
    assert partition.owner_group[-1].item() == -1
    group_zero = report["groups"][partition.owner_group[0].item()]
    assert group_zero == dict(
        group_index=0, cell=[0, 0, 0], core_count=1, ghost_count=1,
        mandatory_neighbor_count=1, mandatory_retained_count=1,
        spatial_neighbor_count=2, spatial_retained_count=0,
        spatial_truncated_count=2,
    )
    group_two = report["groups"][partition.owner_group[2].item()]
    assert group_two["mandatory_neighbor_count"] == 0
    assert group_two["spatial_neighbor_count"] == 2
    assert group_two["spatial_retained_count"] == 1
    assert group_two["spatial_truncated_count"] == 1


def test_numbering_fallback_rejects_ambiguous_duplicate_residue_keys():
    with pytest.raises(ValueError, match="unique.*chain_id, residue_index"):
        build_spatial_partition(
            torch.zeros(2, 3), mask=torch.ones(2, dtype=torch.bool),
            chain_id=torch.zeros(2, dtype=torch.long),
            residue_index=torch.zeros(2, dtype=torch.long),
        )
