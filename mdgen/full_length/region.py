"""Training regions: one random spatial core plus its read-only ghost context.

Interim sampler from #15; #12 replaces it with `group_region.py`, which picks
regions from a stored full-protein partition so they match generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .partition import GroupDiagnostics, _validate_inputs


@dataclass(frozen=True)
class TrainingRegion:
    """Core and ghost tensor indices, each in ascending residue order."""

    center: int
    core_indices: torch.Tensor
    ghost_indices: torch.Tensor
    diagnostics: GroupDiagnostics


def sample_training_region(
    ca_coordinates: torch.Tensor,
    *,
    mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    core_count: int = 256,
    ghost_capacity: int = 256,
    contact_radius: float = 16.0,
    covalent_span: int = 2,
    center: int | None = None,
    generator: torch.Generator | None = None,
) -> TrainingRegion:
    """Sample a region from first-frame C-alpha positions.

    The core is the core_count valid residues nearest a uniformly random valid
    center (ties by index). Ghosts are same-chain residues within covalent_span
    of the core plus residues within contact_radius of any core residue; when
    more than ghost_capacity qualify, covalent neighbors are always kept and the
    farthest spatial ghosts are dropped. Covalent neighbors that do not fit raise.
    """
    _validate_inputs(ca_coordinates, mask, chain_id, residue_index)
    if core_count <= 0:
        raise ValueError("core_count must be positive")
    if ghost_capacity < 0:
        raise ValueError("ghost_capacity must be non-negative")
    if contact_radius <= 0:
        raise ValueError("contact_radius must be positive")
    if covalent_span < 0:
        raise ValueError("covalent_span must be non-negative")

    device = ca_coordinates.device
    ca_cpu = ca_coordinates.detach().float().cpu()
    chain_cpu = chain_id.detach().long().cpu().tolist()
    residue_cpu = residue_index.detach().long().cpu().tolist()
    valid = torch.nonzero(mask.detach().bool().cpu(), as_tuple=False).flatten()
    if valid.numel() == 0:
        raise ValueError("mask has no valid residues")
    lookup = {(chain_cpu[i], residue_cpu[i]): i for i in valid.tolist()}
    if len(lookup) != valid.numel():
        raise ValueError(
            "Valid residues must have unique (chain_id, residue_index) pairs for "
            "the mandatory-neighbor numbering fallback"
        )

    if center is None:
        center = int(valid[torch.randint(valid.numel(), (), generator=generator)])
    elif center not in lookup.values():
        raise ValueError(f"center={center} is not a valid residue")

    to_center = (ca_cpu[valid] - ca_cpu[center]).norm(dim=-1)
    order = sorted(range(valid.numel()), key=lambda i: (float(to_center[i]), int(valid[i])))
    core = sorted(int(valid[i]) for i in order[:core_count])
    core_set = set(core)

    covalent = {
        neighbor
        for index in core
        for delta in range(1, covalent_span + 1)
        for position in (residue_cpu[index] - delta, residue_cpu[index] + delta)
        if (neighbor := lookup.get((chain_cpu[index], position))) is not None
    } - core_set
    if len(covalent) > ghost_capacity:
        raise ValueError(
            f"ghost_capacity={ghost_capacity} cannot retain the {len(covalent)} "
            f"mandatory neighbors of the region around residue {center} "
            f"(covalent_span={covalent_span}); capacity was not widened."
        )

    others = [i for i in valid.tolist() if i not in core_set]
    spatial = []
    if others:
        distances = torch.cdist(ca_cpu[core], ca_cpu[others]).amin(dim=0)
        spatial = [
            (float(distance), index)
            for distance, index in zip(distances.tolist(), others)
            if distance <= contact_radius
        ]
    spatial_indices = {index for _, index in spatial}
    remaining = sorted(item for item in spatial if item[1] not in covalent)
    ghost = sorted(covalent | {
        index for _, index in remaining[: ghost_capacity - len(covalent)]
    })

    return TrainingRegion(
        center=center,
        core_indices=torch.tensor(core, dtype=torch.long, device=device),
        ghost_indices=torch.tensor(ghost, dtype=torch.long, device=device),
        diagnostics=GroupDiagnostics(
            core_count=len(core),
            ghost_count=len(ghost),
            mandatory_neighbor_count=len(covalent),
            mandatory_retained_count=len(covalent),
            spatial_neighbor_count=len(spatial_indices),
            spatial_retained_count=len(spatial_indices.intersection(ghost)),
        ),
    )
