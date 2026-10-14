"""Deterministic core/ghost neighborhoods for full-length proteins."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import torch


@dataclass(frozen=True)
class GroupDiagnostics:
    core_count: int
    ghost_count: int
    mandatory_neighbor_count: int
    mandatory_retained_count: int
    spatial_neighbor_count: int
    spatial_retained_count: int

    @property
    def spatial_truncated_count(self) -> int:
        return self.spatial_neighbor_count - self.spatial_retained_count


@dataclass(frozen=True)
class LocalGroup:
    """One bounded neighborhood with uniquely owned core and read-only ghosts."""

    cell: tuple[int, int, int]
    core_indices: torch.Tensor
    ghost_indices: torch.Tensor
    diagnostics: GroupDiagnostics | None = None

    @property
    def indices(self) -> torch.Tensor:
        return torch.cat((self.core_indices, self.ghost_indices))


@dataclass(frozen=True)
class OwnershipValidation:
    missing_count: int
    duplicate_count: int

    @property
    def valid(self) -> bool:
        return self.missing_count == 0 and self.duplicate_count == 0


@dataclass(frozen=True)
class GatheredGroups:
    values: torch.Tensor
    indices: torch.Tensor
    mask: torch.Tensor
    core_mask: torch.Tensor

    @property
    def ghost_mask(self) -> torch.Tensor:
        return self.mask & ~self.core_mask


@dataclass
class SpatialPartition:
    """A partition of one structure; tensor indices remain on the input device."""

    groups: tuple[LocalGroup, ...]
    owner_group: torch.Tensor
    residue_mask: torch.Tensor
    core_capacity: int
    ghost_capacity: int
    cell_width: float

    def diagnostics(self) -> dict:
        """JSON-ready ownership and neighborhood retention at construction time.

        Spatial neighbors count distinct residues outside the group's core that
        are within contact_radius of any core, including mandatory neighbors.
        """
        ownership = self.validate_ownership()
        groups = []
        for index, group in enumerate(self.groups):
            entry = {
                "group_index": index,
                "cell": list(group.cell),
                "core_count": group.core_indices.numel(),
                "ghost_count": group.ghost_indices.numel(),
            }
            if group.diagnostics is not None:
                details = group.diagnostics
                entry.update({
                    "mandatory_neighbor_count": details.mandatory_neighbor_count,
                    "mandatory_retained_count": details.mandatory_retained_count,
                    "spatial_neighbor_count": details.spatial_neighbor_count,
                    "spatial_retained_count": details.spatial_retained_count,
                    "spatial_truncated_count": details.spatial_truncated_count,
                })
            groups.append(entry)
        owner_group_matches = all(
            bool((self.owner_group[group.core_indices] == index).all().item())
            for index, group in enumerate(self.groups)
        ) and bool((self.owner_group[~self.residue_mask.bool()] == -1).all().item())
        return {
            "ownership_valid": ownership.valid and owner_group_matches,
            "missing_count": ownership.missing_count,
            "duplicate_count": ownership.duplicate_count,
            "owner_group_matches": owner_group_matches,
            "valid_residue_count": int(self.residue_mask.sum().item()),
            "group_count": len(self.groups),
            "groups": groups,
        }

    def validate_ownership(self) -> OwnershipValidation:
        counts = torch.zeros_like(self.owner_group)
        for group in self.groups:
            counts[group.core_indices] += 1
        expected = self.residue_mask.bool()
        return OwnershipValidation(
            missing_count=int((expected & (counts == 0)).sum().item()),
            duplicate_count=int((counts > 1).sum().item()),
        )

    def gather(self, values: torch.Tensor) -> GatheredGroups:
        if values.shape[0] != self.residue_mask.shape[0]:
            raise ValueError("The leading values dimension must match the partition")
        width = max((group.indices.numel() for group in self.groups), default=0)
        indices = torch.full(
            (len(self.groups), width), -1, dtype=torch.long, device=values.device
        )
        mask = torch.zeros((len(self.groups), width), dtype=torch.bool, device=values.device)
        core_mask = torch.zeros_like(mask)
        output = values.new_zeros((len(self.groups), width, *values.shape[1:]))
        for group_index, group in enumerate(self.groups):
            group_indices = group.indices.to(values.device)
            count = group_indices.numel()
            core_count = group.core_indices.numel()
            indices[group_index, :count] = group_indices
            mask[group_index, :count] = True
            core_mask[group_index, :core_count] = True
            output[group_index, :count] = values[group_indices]
        return GatheredGroups(output, indices, mask, core_mask)

    def scatter_core(
        self, local_values: torch.Tensor, gathered: GatheredGroups | None = None
    ) -> torch.Tensor:
        if gathered is None:
            raise ValueError("scatter_core requires the GatheredGroups metadata")
        if local_values.shape[:2] != gathered.values.shape[:2]:
            raise ValueError("Local values do not match gathered group dimensions")
        output = local_values.new_zeros(
            (self.residue_mask.shape[0], *local_values.shape[2:])
        )
        core_indices = gathered.indices[gathered.core_mask]
        output[core_indices] = local_values[gathered.core_mask]
        return output

    def ghost_update_violations(
        self, local_values: torch.Tensor, gathered: GatheredGroups
    ) -> int:
        """Return violations of the assembly contract (ghosts are never scattered)."""
        assembled = self.scatter_core(local_values, gathered)
        ghost_only = local_values.new_zeros(local_values.shape)
        ghost_only[gathered.ghost_mask] = local_values[gathered.ghost_mask]
        ghost_assembled = self.scatter_core(ghost_only, gathered)
        return int(torch.count_nonzero(ghost_assembled).item()) + int(
            not torch.isfinite(assembled).all().item()
        )


def _validate_inputs(
    ca_coordinates: torch.Tensor,
    mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
) -> None:
    residues = ca_coordinates.shape[0]
    if ca_coordinates.shape != (residues, 3):
        raise ValueError("ca_coordinates must have shape [L, 3]")
    for name, tensor in (
        ("mask", mask),
        ("chain_id", chain_id),
        ("residue_index", residue_index),
    ):
        if tensor.shape != (residues,):
            raise ValueError(f"{name} must have shape [L]")


def build_spatial_partition(
    ca_coordinates: torch.Tensor,
    *,
    mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    cell_width: float = 16.0,
    contact_radius: float = 16.0,
    core_capacity: int = 128,
    ghost_capacity: int = 128,
    covalent_span: int = 2,
) -> SpatialPartition:
    """Build fixed neighborhoods from conditioned first-frame C-alpha positions.

    Mandatory neighbors use same-chain residue numbering within covalent_span;
    this is a numbering fallback, not a topology-based bond lookup. Overfull
    cores are split deterministically rather than dropping mandatory ghosts.
    """
    _validate_inputs(ca_coordinates, mask, chain_id, residue_index)
    if cell_width <= 0 or contact_radius <= 0:
        raise ValueError("cell_width and contact_radius must be positive")
    if core_capacity <= 0 or ghost_capacity < 0:
        raise ValueError("capacities must be non-negative and core_capacity positive")
    if covalent_span < 0:
        raise ValueError("covalent_span must be non-negative")

    device = ca_coordinates.device
    ca_cpu = ca_coordinates.detach().float().cpu()
    mask_cpu = mask.detach().bool().cpu()
    chain_cpu = chain_id.detach().long().cpu()
    residue_cpu = residue_index.detach().long().cpu()
    valid_list = torch.nonzero(mask_cpu, as_tuple=False).flatten().tolist()
    cell_coords = torch.floor(ca_cpu / cell_width).to(torch.long)
    cells: dict[tuple[int, int, int], list[int]] = {}
    for index in valid_list:
        cell = tuple(int(v) for v in cell_coords[index].tolist())
        cells.setdefault(cell, []).append(index)
    covalent_lookup = {
        (int(chain_cpu[index]), int(residue_cpu[index])): index for index in valid_list
    }
    if len(covalent_lookup) != len(valid_list):
        raise ValueError(
            "Valid residues must have unique (chain_id, residue_index) pairs for "
            "the mandatory-neighbor numbering fallback"
        )

    owner_group = torch.full((mask.numel(),), -1, dtype=torch.long, device=device)
    groups: list[LocalGroup] = []
    cell_reach = max(1, math.ceil(contact_radius / cell_width))
    neighbor_offsets = tuple(product(range(-cell_reach, cell_reach + 1), repeat=3))

    def mandatory_neighbors(core_list: list[int]) -> set[int]:
        covalent: set[int] = set()
        for core_index in core_list:
            chain = int(chain_cpu[core_index])
            sequence_position = int(residue_cpu[core_index])
            for delta in range(1, covalent_span + 1):
                for neighbor_position in (
                    sequence_position - delta,
                    sequence_position + delta,
                ):
                    neighbor = covalent_lookup.get((chain, neighbor_position))
                    if neighbor is not None:
                        covalent.add(neighbor)
        return covalent.difference(core_list)

    def split_core(core_list: list[int]) -> list[list[int]]:
        mandatory_count = len(mandatory_neighbors(core_list))
        if mandatory_count <= ghost_capacity:
            return [core_list]
        if len(core_list) == 1:
            index = core_list[0]
            raise ValueError(
                f"ghost_capacity={ghost_capacity} cannot retain the "
                f"{mandatory_count} mandatory neighbors of residue tensor index "
                f"{index} (chain_id={int(chain_cpu[index])}, "
                f"residue_index={int(residue_cpu[index])}, "
                f"covalent_span={covalent_span}). Increase ghost_capacity or "
                "check the chain/residue numbering; capacity was not widened."
            )
        middle = len(core_list) // 2
        return split_core(core_list[:middle]) + split_core(core_list[middle:])

    for cell in sorted(cells):
        cell_members = cells[cell]
        core_lists = [
            split_group
            for start in range(0, len(cell_members), core_capacity)
            for split_group in split_core(cell_members[start : start + core_capacity])
        ]
        for core_list in core_lists:
            core = torch.tensor(core_list, dtype=torch.long, device=device)
            core_set = set(core_list)
            covalent = mandatory_neighbors(core_list)

            spatial_candidates: set[int] = set()
            for offset in neighbor_offsets:
                nearby_cell = tuple(cell[axis] + offset[axis] for axis in range(3))
                spatial_candidates.update(cells.get(nearby_cell, ()))
            spatial_candidates.difference_update(core_set)
            if spatial_candidates:
                candidates = torch.tensor(
                    sorted(spatial_candidates), dtype=torch.long, device=device
                )
                distances = torch.cdist(
                    ca_cpu[core_list], ca_cpu[candidates.cpu()]
                ).amin(dim=0)
                spatial = [
                    (float(distance), int(index))
                    for distance, index in zip(distances.tolist(), candidates.tolist())
                    if distance <= contact_radius
                ]
            else:
                spatial = []

            priority = sorted(
                covalent,
                key=lambda index: (
                    min(
                        abs(int(residue_cpu[index]) - int(residue_cpu[c]))
                        for c in core_list
                        if int(chain_cpu[index]) == int(chain_cpu[c])
                    ),
                    index,
                ),
            )
            remaining = sorted(
                ((distance, index) for distance, index in spatial if index not in covalent),
                key=lambda item: (item[0], item[1]),
            )
            ghost_list = priority + [
                index for _, index in remaining[: ghost_capacity - len(priority)]
            ]
            ghost = torch.tensor(ghost_list, dtype=torch.long, device=device)
            spatial_indices = {index for _, index in spatial}
            details = GroupDiagnostics(
                core_count=len(core_list),
                ghost_count=len(ghost_list),
                mandatory_neighbor_count=len(covalent),
                mandatory_retained_count=len(covalent.intersection(ghost_list)),
                spatial_neighbor_count=len(spatial_indices),
                spatial_retained_count=len(spatial_indices.intersection(ghost_list)),
            )

            group_index = len(groups)
            owner_group[core] = group_index
            groups.append(LocalGroup(
                cell=cell, core_indices=core, ghost_indices=ghost, diagnostics=details
            ))

    partition = SpatialPartition(
        groups=tuple(groups),
        owner_group=owner_group,
        residue_mask=mask.bool(),
        core_capacity=core_capacity,
        ghost_capacity=ghost_capacity,
        cell_width=cell_width,
    )
    validation = partition.validate_ownership()
    if not validation.valid:
        raise RuntimeError(f"Invalid core ownership: {validation}")
    return partition
