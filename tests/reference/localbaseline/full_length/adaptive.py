"""Window-boundary-only refresh for fixed spatial neighborhoods."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .partition import SpatialPartition, build_spatial_partition


@dataclass(frozen=True)
class RepartitionEvent:
    frame: int
    trigger_displacement: float
    ownership_valid: bool
    state_roundtrip_max_error: float


class AdaptiveSpatialPartitioner:
    def __init__(
        self,
        initial_coordinates: torch.Tensor,
        *,
        mask: torch.Tensor,
        chain_id: torch.Tensor,
        residue_index: torch.Tensor,
        displacement_threshold: float = 8.0,
        **partition_options,
    ):
        self.mask = mask.bool()
        self.chain_id = chain_id
        self.residue_index = residue_index
        self.displacement_threshold = displacement_threshold
        self.partition_options = partition_options
        self.reference_coordinates = initial_coordinates.detach().clone()
        self.partition = self._build(initial_coordinates)
        self.events: list[RepartitionEvent] = []

    def _build(self, coordinates: torch.Tensor) -> SpatialPartition:
        return build_spatial_partition(
            coordinates,
            mask=self.mask,
            chain_id=self.chain_id,
            residue_index=self.residue_index,
            **self.partition_options,
        )

    def maybe_repartition(
        self, coordinates: torch.Tensor, *, frame: int, window_boundary: bool
    ) -> bool:
        displacement = torch.linalg.vector_norm(
            coordinates - self.reference_coordinates, dim=-1
        )
        trigger = float(displacement[self.mask].max().item()) if self.mask.any() else 0.0
        if not window_boundary or trigger < self.displacement_threshold:
            return False

        new_partition = self._build(coordinates)
        gathered = new_partition.gather(coordinates)
        restored = new_partition.scatter_core(gathered.values, gathered)
        error = float((restored[self.mask] - coordinates[self.mask]).abs().max().item())
        validation = new_partition.validate_ownership()
        self.partition = new_partition
        self.reference_coordinates = coordinates.detach().clone()
        self.events.append(
            RepartitionEvent(
                frame=frame,
                trigger_displacement=trigger,
                ownership_valid=validation.valid,
                state_roundtrip_max_error=error,
            )
        )
        return True
