"""Region sampling from a stored full-protein partition, and full-protein RoPE (#12 Ticket 3)."""
import copy

import pytest
import torch

from mdgen.full_length.group_region import region_subpartition, sample_region_groups
from mdgen.full_length.partition import build_spatial_partition
from mdgen.full_length.attention import run_pretrained_local_attention

L = 120


def protein(seed=0):
    generator = torch.Generator().manual_seed(seed)
    step = torch.randn(L, 3, generator=generator)
    ca = (3.8 * step / step.norm(dim=-1, keepdim=True)).cumsum(0)
    chain_id = (torch.arange(L) >= L // 2).long()
    residue_index = torch.arange(L) % (L // 2)
    partition = build_spatial_partition(
        ca, mask=torch.ones(L, dtype=torch.bool), chain_id=chain_id,
        residue_index=residue_index, cell_width=8.0, contact_radius=8.0,
        core_capacity=8, ghost_capacity=64)
    return ca, partition


def sample(partition, ca, budget, halo_rings=1, seed=0):
    return sample_region_groups(partition, ca, region_residues=budget, halo_rings=halo_rings,
                                generator=torch.Generator().manual_seed(seed))


def test_region_respects_the_budget_and_stops_at_the_first_overflow():
    ca, partition = protein()
    selection = sample(partition, ca, budget=60)
    assert len(selection.residues) <= 60
    # adding the next-nearest group would have exceeded the budget
    assert selection.next_region_size is None or selection.next_region_size > 60
    assert len(selection.loss_groups) >= 1


def test_each_loss_residue_belongs_to_exactly_one_loss_group():
    ca, partition = protein()
    selection = sample(partition, ca, budget=80)
    loss_cores = torch.cat([partition.groups[g].core_indices for g in selection.loss_groups])
    assert len(loss_cores) == len(set(loss_cores.tolist()))
    sub = region_subpartition(partition, selection)
    assert int(sub.loss_core_mask.sum()) == len(loss_cores)
    assert sub.residues[sub.loss_core_mask].tolist() == sorted(loss_cores.tolist())


def owners_of_ghosts(partition, groups):
    ghosts = torch.cat([partition.groups[g].ghost_indices for g in groups])
    return set(partition.owner_group[ghosts].tolist())


@pytest.mark.parametrize('rings', [0, 1, 2])
def test_halo_rings_are_owners_of_the_previous_rings_ghosts(rings):
    ca, partition = protein()
    selection = sample(partition, ca, budget=90, halo_rings=rings)
    selected, frontier, expected = set(selection.loss_groups), selection.loss_groups, set()
    for _ in range(rings):
        frontier = owners_of_ghosts(partition, frontier) - selected
        selected |= frontier
        expected |= frontier
    assert set(selection.halo_groups) == expected
    assert not set(selection.loss_groups) & set(selection.halo_groups)
    if rings:
        assert selection.halo_groups  # a 90-residue region of a 120-residue protein has a halo


def test_context_residues_have_no_owner_and_no_loss():
    ca, partition = protein()
    selection = sample(partition, ca, budget=60)
    sub = region_subpartition(partition, selection)
    owned = torch.cat([group.core_indices for group in sub.partition.groups])
    context = torch.ones(len(sub.residues), dtype=torch.bool)
    context[owned] = False
    assert context.any()
    assert (sub.partition.owner_group[context] == -1).all()
    assert not sub.loss_core_mask[context].any()
    # context residues are not "valid" owned residues: ownership checks pass, summaries skip them
    assert torch.equal(sub.partition.residue_mask, ~context)
    assert sub.partition.validate_ownership().valid
    assert sub.partition.diagnostics()['ownership_valid']
    # every ghost of every selected group is inside the region
    for group in sub.partition.groups:
        assert group.ghost_indices.max() < len(sub.residues)


def test_same_seed_gives_the_same_region():
    ca, partition = protein()
    first, again = sample(partition, ca, 60, seed=4), sample(partition, ca, 60, seed=4)
    assert first.loss_groups == again.loss_groups and torch.equal(first.residues, again.residues)


def test_seed_group_larger_than_the_budget_raises():
    ca, partition = protein()
    with pytest.raises(ValueError, match='region_residues'):
        sample(partition, ca, budget=4)


def test_whole_protein_region_reproduces_the_full_partition():
    ca, partition = protein()
    sub = region_subpartition(partition, sample(partition, ca, budget=L))
    assert torch.equal(sub.residues, torch.arange(L))
    assert sub.loss_core_mask.all()
    assert len(sub.partition.groups) == len(partition.groups)
    for a, b in zip(sub.partition.groups, partition.groups):
        assert torch.equal(a.core_indices, b.core_indices)
        assert torch.equal(a.ghost_indices, b.ghost_indices)
    assert torch.equal(sub.partition.owner_group, partition.owner_group)
    assert torch.equal(sub.partition.position_index, torch.arange(L))


def local_model():
    from types import SimpleNamespace
    from mdgen.full_length.model import LocalLatentMDGenModel
    args = SimpleNamespace(design=False, tps_condition=False, inpainting=False,
        dynamic_mpnn=False, mpnn=False, embed_dim=32, ipa_head_dim=4, ipa_heads=2,
        ipa_qk=2, ipa_v=2, dropout=0., prepend_ipa=True, interleave_ipa=True,
        no_aa_emb=False, num_layers=1, mha_heads=4, no_rope=False, hyena=False,
        num_frames=3, abs_pos_emb=False, abs_time_emb=False, grad_checkpointing=False,
        time_multiplier=100., sim_condition=True, local_spatial=True, temporal_sdpa=True,
        cell_summaries=False, group_batch_size=2, spatial_cell_width=8.,
        spatial_contact_radius=8., core_capacity=8, ghost_capacity=64, ffn_chunk_size=65536)
    torch.manual_seed(0)
    model = LocalLatentMDGenModel(args, 21)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.08)
    return model.eval()


def test_region_attention_uses_full_protein_positions():
    ca, partition = protein()
    sub = region_subpartition(partition, sample(partition, ca, budget=60))
    attention = local_model().layers[0].mha_l
    states = torch.randn(2, L, 32, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        full = run_pretrained_local_attention(attention, states, partition)
        region = run_pretrained_local_attention(
            attention, states[:, sub.residues], sub.partition)
        renumbered = copy.copy(sub.partition)
        renumbered.position_index, renumbered.full_length = None, None
        local_slots = run_pretrained_local_attention(
            attention, states[:, sub.residues], renumbered)
    loss = sub.loss_core_mask
    assert (sub.residues.diff() > 1).any()  # not one contiguous block, where slots would also work
    # a loss core sees exactly its full-protein ghosts, so its first attention update matches
    torch.testing.assert_close(region[:, loss], full[:, sub.residues[loss]], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(local_slots[:, loss], full[:, sub.residues[loss]], atol=1e-4)


def test_whole_protein_region_gives_the_same_model_output():
    from mdgen.rigid_utils import Rigid
    ca, partition = protein()
    sub = region_subpartition(partition, sample(partition, ca, budget=L))
    model = local_model()
    frames = Rigid(Rigid.identity((1, L)).get_rots(), ca[None])
    kwargs = dict(mask=torch.ones(1, 3, L), start_frames=frames, aatype=torch.zeros(1, L).long(),
                  chain_id=(torch.arange(L) >= L // 2).long()[None],
                  residue_index=(torch.arange(L) % (L // 2))[None])
    x, t = torch.randn(1, 3, L, 21, generator=torch.Generator().manual_seed(2)), torch.tensor([0.4])
    with torch.no_grad():
        full = model.forward_inference(x, t, spatial_partitions=[partition], **kwargs)
        region = model.forward_inference(x, t, spatial_partitions=[sub.partition], **kwargs)
    torch.testing.assert_close(region, full)


def test_cell_summaries_leave_context_residues_unchanged():
    from mdgen.full_length.summaries import CoreCellSummaryExchange
    ca, partition = protein()
    sub = region_subpartition(partition, sample(partition, ca, budget=60))
    exchange = CoreCellSummaryExchange(32, 4)
    with torch.no_grad():  # the injection is zero-initialised; make it act
        for parameter in exchange.parameters():
            parameter.normal_(0, 0.5)
    states = torch.randn(2, len(sub.residues), 32)
    with torch.no_grad():
        out = exchange(states, sub.partition)
    context = sub.partition.owner_group == -1
    assert context.any()
    torch.testing.assert_close(out[:, context], states[:, context])


def test_ca_length_must_match_the_partition():
    ca, partition = protein()
    with pytest.raises(ValueError, match='ca_coordinates'):
        sample(partition, ca[:-1], budget=60)


def test_region_positions_need_the_full_length():
    from mdgen.full_length.partition import SpatialPartition
    ca, partition = protein()
    with pytest.raises(ValueError, match='full_length'):
        SpatialPartition(partition.groups, partition.owner_group, partition.residue_mask,
                         8, 64, 8.0, position_index=torch.arange(L))
