"""Integrated bounded spatial/temporal block used by scale validation."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..model.ipa import InvariantPointAttention
from ..rigid_utils import Rigid
from .attention import LocalSelfAttention
from .local_ipa import LocalInvariantPointAttention
from .partition import SpatialPartition
from .summaries import CoreCellSummaryExchange
from .temporal import TemporalMixer


class BoundedTrajectoryBlock(nn.Module):
    """One local/global spatial layer followed by full-frame Hyena mixing."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_frames: int,
        attention_backend: str = "sdpa",
    ):
        super().__init__()
        self.spatial = LocalSelfAttention(embed_dim, num_heads, attention_backend)
        self.spatial_norm = nn.LayerNorm(embed_dim)
        self.summaries = CoreCellSummaryExchange(embed_dim, num_heads)
        ipa = InvariantPointAttention(
            c_s=embed_dim,
            c_z=0,
            c_hidden=max(1, embed_dim // num_heads),
            no_heads=num_heads,
            no_qk_points=1,
            no_v_points=1,
        )
        self.local_ipa = LocalInvariantPointAttention(ipa)
        self.ipa_norm = nn.LayerNorm(embed_dim)
        self.temporal = TemporalMixer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_frames=num_frames,
            backend="hyena",
        )
        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.coordinate_head = nn.Linear(embed_dim, 3)

    @staticmethod
    def _map_frame_chunks(
        function,
        value: torch.Tensor,
        frame_batch_size: int | None,
    ) -> torch.Tensor:
        if frame_batch_size is None:
            frame_batch_size = value.shape[0]
        if frame_batch_size <= 0:
            raise ValueError("temporal_batch_size must be positive")
        outputs = []
        for start in range(0, value.shape[0], frame_batch_size):
            stop = min(value.shape[0], start + frame_batch_size)
            chunk = value[start:stop]
            if chunk.requires_grad:
                outputs.append(
                    checkpoint(
                        lambda current, first=start, last=stop: function(
                            current, first, last
                        ),
                        chunk,
                        use_reentrant=False,
                    )
                )
            else:
                outputs.append(function(chunk, start, stop))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        states: torch.Tensor,
        frames: Rigid,
        partition: SpatialPartition,
        *,
        group_batch_size: int,
        temporal_batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states.shape[0] != 1:
            raise ValueError("Scale-validation block currently accepts one molecular system")
        trajectory = states[0]
        trajectory = trajectory + self._map_frame_chunks(
            lambda value, _start, _stop: self.spatial(
                self.spatial_norm(value),
                partition,
                group_batch_size=group_batch_size,
            ),
            trajectory,
            temporal_batch_size,
        )
        trajectory = self._map_frame_chunks(
            lambda value, _start, _stop: self.summaries(value, partition),
            trajectory,
            temporal_batch_size,
        )
        trajectory = trajectory + self._map_frame_chunks(
            lambda value, start, stop: self.local_ipa(
                self.ipa_norm(value),
                frames[start:stop],
                partition,
                group_batch_size=group_batch_size,
            ),
            trajectory,
            temporal_batch_size,
        )
        states = trajectory.unsqueeze(0)
        states = states + self.temporal(
            self.temporal_norm(states),
            sequence_batch_size=temporal_batch_size,
            checkpoint_chunks=states.requires_grad,
        )
        displacement = 0.01 * self.coordinate_head(states).unsqueeze(-2)
        return states, displacement
