"""Bounded adapter for the existing invariant point attention module."""

from __future__ import annotations

import torch
from torch import nn

from ..rigid_utils import Rigid
from .partition import SpatialPartition


class LocalInvariantPointAttention(nn.Module):
    """Run IPA inside core/ghost groups and scatter owned outputs only."""

    def __init__(self, ipa: nn.Module):
        super().__init__()
        self.ipa = ipa

    def forward(
        self,
        states: torch.Tensor,
        frames: Rigid,
        partition: SpatialPartition,
        *,
        group_batch_size: int = 1,
    ) -> torch.Tensor:
        return run_local_ipa(
            self.ipa,
            states,
            frames,
            partition,
            group_batch_size=group_batch_size,
        )


def run_local_ipa(
    ipa: nn.Module,
    states: torch.Tensor,
    frames: Rigid,
    partition: SpatialPartition,
    *,
    group_batch_size: int = 1,
) -> torch.Tensor:
    """Apply an existing IPA module to bounded groups without wrapping it per call."""
    if states.ndim != 3 or states.shape[1] != partition.residue_mask.numel():
        raise ValueError("states must have shape [N, L, C] for this partition")
    if frames.shape != states.shape[:2]:
        raise ValueError("frames must have shape [N, L]")
    if group_batch_size <= 0:
        raise ValueError("group_batch_size must be positive")

    gathered = partition.gather(states.transpose(0, 1))
    gathered_frames = partition.gather(frames.to_tensor_7().transpose(0, 1))
    grouped = gathered.values.permute(2, 0, 1, 3)
    grouped_frames = gathered_frames.values.permute(2, 0, 1, 3)
    outputs = torch.zeros_like(grouped)

    for start in range(0, len(partition.groups), group_batch_size):
        stop = min(len(partition.groups), start + group_batch_size)
        state_chunk = grouped[:, start:stop]
        frame_chunk = grouped_frames[:, start:stop]
        flat_states = state_chunk.reshape(
            -1, state_chunk.shape[-2], state_chunk.shape[-1]
        )
        flat_frames = Rigid.from_tensor_7(
            frame_chunk.reshape(-1, frame_chunk.shape[-2], 7)
        )
        flat_mask = gathered.mask[start:stop].to(states.dtype).repeat(
            states.shape[0], 1
        )
        outputs[:, start:stop] = ipa(
            flat_states, flat_frames, frame_mask=flat_mask
        ).reshape_as(state_chunk)

    local_outputs = outputs.permute(1, 2, 0, 3)
    assembled = partition.scatter_core(local_outputs, gathered)
    return assembled.permute(1, 0, 2)
