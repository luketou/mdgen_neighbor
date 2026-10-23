"""Training regions chosen from a stored full-protein partition (#12).

A region is a set of loss groups (nearest to a random seed group), the halo
groups owning their ghosts (`halo_rings` deep), and every ghost of those
groups. Halo groups run normally but are not in the loss; ghosts of halo
groups that no selected group owns are context residues: keys only, with
`owner_group == -1` and `residue_mask` False, so their spatial state is not
updated (an accepted approximation).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .partition import LocalGroup, SpatialPartition


@dataclass(frozen=True)
class RegionSelection:
    loss_groups: tuple[int, ...]
    halo_groups: tuple[int, ...]
    residues: torch.Tensor  # full-protein tensor indices, ascending
    next_region_size: int | None  # size had the next-nearest group been added


@dataclass(frozen=True)
class RegionPartition:
    partition: SpatialPartition  # indexed by region position; carries position_index
    loss_core_mask: torch.Tensor  # region positions owned by a loss group

    @property
    def residues(self):
        """Full-protein index of each region position."""
        return self.partition.position_index


def _halo(partition, loss_groups, rings):
    owner = partition.owner_group.cpu()
    selected, frontier, halo = set(loss_groups), list(loss_groups), []
    for _ in range(rings):
        ghosts = torch.cat([partition.groups[g].ghost_indices.cpu() for g in frontier])
        frontier = sorted(set(owner[ghosts].tolist()) - selected - {-1})
        if not frontier:
            break
        selected.update(frontier)
        halo.extend(frontier)
    return tuple(sorted(halo))


def _residues(partition, groups):
    return torch.unique(torch.cat([partition.groups[g].indices.cpu() for g in groups]))


def sample_region_groups(partition, ca_coordinates, *, region_residues, halo_rings=1,
                         generator=None):
    """Grow loss groups around a random seed group until the next would exceed the budget.

    Groups are added by distance between core C-alpha centres (ties by index);
    region_residues counts every residue in the region tensor.
    """
    if halo_rings < 0:
        raise ValueError('halo_rings must be non-negative')
    if not partition.groups:
        raise ValueError('partition has no groups')
    ca = torch.as_tensor(ca_coordinates).detach().float().cpu()
    if ca.shape != (partition.residue_mask.numel(), 3):
        raise ValueError(f'ca_coordinates must have shape [{partition.residue_mask.numel()}, 3] '
                         'for this full-protein partition')
    centres = torch.stack([ca[g.core_indices.cpu()].mean(0) for g in partition.groups])
    seed = int(torch.randint(len(partition.groups), (), generator=generator))
    distance = (centres - centres[seed]).norm(dim=-1)
    order = sorted(range(len(partition.groups)), key=lambda g: (float(distance[g]), g))

    loss, halo, residues, next_size = [], (), None, None
    for group in order:
        trial = loss + [group]
        trial_halo = _halo(partition, trial, halo_rings)
        trial_residues = _residues(partition, trial + list(trial_halo))
        if len(trial_residues) > region_residues:
            if not loss:
                raise ValueError(
                    f'region_residues={region_residues} cannot hold seed group {seed} '
                    f'with {halo_rings} halo ring(s) ({len(trial_residues)} residues)')
            next_size = len(trial_residues)
            break
        loss, halo, residues = trial, trial_halo, trial_residues
    return RegionSelection(tuple(sorted(loss)), halo, residues, next_size)


def largest_seed_region(partition, halo_rings=1):
    """Residues in the smallest region each group can seed; the largest over all groups.

    A region_residues budget below this value fails for some seed group.
    """
    return max(len(_residues(partition, [g, *_halo(partition, [g], halo_rings)]))
               for g in range(len(partition.groups)))


def region_subpartition(partition, selection):
    """Re-index the selected groups to the region tensor (ascending full-protein order)."""
    residues = selection.residues
    full_to_region = torch.full((partition.residue_mask.numel(),), -1, dtype=torch.long)
    full_to_region[residues] = torch.arange(len(residues))
    group_ids = sorted(selection.loss_groups + selection.halo_groups)
    owner_group = torch.full((len(residues),), -1, dtype=torch.long)
    groups = []
    for new_index, group_id in enumerate(group_ids):
        group = partition.groups[group_id]
        core = full_to_region[group.core_indices.cpu()]
        ghost = full_to_region[group.ghost_indices.cpu()]
        if (core < 0).any() or (ghost < 0).any():
            raise ValueError(f'group {group_id} has residues not in the region selection')
        owner_group[core] = new_index
        groups.append(LocalGroup(group.cell, core, ghost, group.diagnostics))
    loss_cores = torch.cat([partition.groups[g].core_indices.cpu() for g in selection.loss_groups])
    position_index, full_length = residues, partition.residue_mask.numel()
    if partition.position_index is not None:  # a region of a region: keep full-protein positions
        position_index = partition.position_index.cpu()[residues]
        full_length = partition.full_length
    return RegionPartition(
        partition=SpatialPartition(
            groups=tuple(groups), owner_group=owner_group,
            residue_mask=owner_group >= 0,  # context residues are keys only
            core_capacity=partition.core_capacity, ghost_capacity=partition.ghost_capacity,
            cell_width=partition.cell_width, position_index=position_index,
            full_length=full_length),
        loss_core_mask=torch.isin(residues, loss_cores),  # in the source partition's indexing
    )
