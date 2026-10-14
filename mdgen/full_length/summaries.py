"""Core-only cell summaries for bounded long-range exchange."""

from __future__ import annotations

import torch
from torch import nn

from .partition import SpatialPartition


class CoreCellSummaryExchange(nn.Module):
    """Pool owned residues, exchange across cells, and inject owner context."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.summary_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.summary_norm = nn.LayerNorm(embed_dim)
        self.inject = nn.Linear(embed_dim, embed_dim)
        self.gate = nn.Parameter(torch.zeros(()))

    @staticmethod
    def pool_core(grouped: torch.Tensor, core_mask: torch.Tensor) -> torch.Tensor:
        if grouped.ndim != 4:
            raise ValueError("grouped values must have shape [N, G, Q, C]")
        weights = core_mask.to(grouped.dtype)[None, :, :, None]
        return (grouped * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)

    def forward(self, states: torch.Tensor, partition: SpatialPartition) -> torch.Tensor:
        if states.ndim != 3 or states.shape[1] != partition.residue_mask.numel():
            raise ValueError("states must have shape [N, L, C] for this partition")
        if not partition.groups:
            return states
        gathered = partition.gather(states.transpose(0, 1))
        grouped = gathered.values.permute(2, 0, 1, 3)
        summaries = self.pool_core(grouped, gathered.core_mask)
        exchanged, _ = self.summary_attention(
            summaries,
            summaries,
            summaries,
            need_weights=False,
        )
        exchanged = self.summary_norm(summaries + exchanged)

        owner = partition.owner_group.to(states.device)
        valid = partition.residue_mask.to(states.device).bool()
        context = torch.zeros_like(states)
        context[:, valid] = exchanged[:, owner[valid]]
        injected = self.inject(context) * valid[None, :, None]
        return states + self.gate * injected
