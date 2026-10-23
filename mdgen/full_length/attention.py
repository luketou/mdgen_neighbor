"""Replaceable dense-in-group attention with core-only assembly."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.nn import functional as F

from .partition import SpatialPartition


def _pretrained_group_attention(attention, states, valid_mask, position_ids, full_length, query_width=None):
    """Use the dense module's parameters, scale and global RoPE positions."""
    batch, length, channels = states.shape
    query_width = length if query_width is None else query_width
    q = attention.q_proj(states[:, :query_width]) * attention.scaling
    k = attention.k_proj(states)
    v = attention.v_proj(states)
    key_positions = position_ids
    if attention.bias_k is not None:
        k = torch.cat((k, attention.bias_k.expand(batch, -1, -1)), dim=1)
        v = torch.cat((v, attention.bias_v.expand(batch, -1, -1)), dim=1)
        valid_mask = F.pad(valid_mask, (0, 1), value=True)
        key_positions = F.pad(key_positions, (0, 1), value=full_length)
    if attention.add_zero_attn:
        k = torch.cat((k, k.new_zeros(batch, 1, channels)), dim=1)
        v = torch.cat((v, v.new_zeros(batch, 1, channels)), dim=1)
        valid_mask = F.pad(valid_mask, (0, 1), value=True)
        key_positions = F.pad(
            key_positions, (0, 1), value=full_length + int(attention.bias_k is not None)
        )
    heads, head_dim = attention.num_heads, attention.head_dim
    q = q.reshape(batch, query_width, heads, head_dim).transpose(1, 2)
    k = k.reshape(batch, -1, heads, head_dim).transpose(1, 2)
    v = v.reshape(batch, -1, heads, head_dim).transpose(1, 2)
    if attention.rot_emb is not None:
        # Rotating at local slot numbers would change residue distances when
        # ghosts are reordered. Bias/zero KV keep their dense sequence slots.
        def rotate(value, positions):
            frequencies = positions.to(attention.rot_emb.inv_freq.dtype)[..., None] * attention.rot_emb.inv_freq
            angles = torch.cat((frequencies, frequencies), dim=-1)[:, None]
            first, second = value.chunk(2, dim=-1)
            rotated = value * angles.cos() + torch.cat((-second, first), dim=-1) * angles.sin()
            # ESM's float32 frequency buffer promotes bf16 projections. SDPA
            # requires Q/K/V to share dtype; restore their projection dtype.
            return rotated.to(value.dtype)

        q = rotate(q, position_ids[:, :query_width])
        k = rotate(k, key_positions)
    attended = F.scaled_dot_product_attention(
        q, k, v, attn_mask=valid_mask[:, None, None, :],
        dropout_p=attention.dropout if attention.training else 0.0,
        scale=1.0,  # q is already scaled by the pretrained module.
    )
    return attention.out_proj(attended.transpose(1, 2).reshape(batch, query_width, channels))


def run_pretrained_local_attention(
    module, states: torch.Tensor, partition: SpatialPartition, *,
    mask: torch.Tensor | None = None, group_batch_size: int = 1,
    backend: str = "sdpa",
) -> torch.Tensor:
    """Localize an existing AttentionWithRoPE without registering new weights.

    Only core queries/outputs are evaluated; ghosts supply key/value context
    and their inputs still receive gradients. Groups are gathered in bounded
    batches, with padding determined by the current batch. This
    adapter supports the model's ordinary self-attention call, without caches,
    pairwise biases or cross-attention.
    """
    if backend != "sdpa":
        raise ValueError("Pretrained local attention currently supports only sdpa")
    attention = module.attn
    if not attention.qkv_same_dim or attention.onnx_trace:
        raise ValueError("Pretrained local attention requires same-dimension self-attention")
    if states.ndim != 3 or states.shape[1] != partition.residue_mask.numel():
        raise ValueError("states must have shape [N, L, C] for this partition")
    if group_batch_size <= 0:
        raise ValueError("group_batch_size must be positive")
    if mask is not None and mask.shape != states.shape[:2]:
        raise ValueError("mask must have shape [N, L]")
    if not partition.groups:
        return torch.zeros_like(states)
    assembled = torch.zeros_like(states)
    for start in range(0, len(partition.groups), group_batch_size):
        stop = min(len(partition.groups), start + group_batch_size)
        # Gather only this bounded group batch; its padding width is local to
        # the batch rather than the largest neighborhood in the full protein.
        selected = partition.groups[start:stop]
        gathered = replace(partition, groups=selected).gather(states.transpose(0, 1))
        chunk = gathered.values.permute(2, 0, 1, 3)
        flat = chunk.reshape(-1, chunk.shape[-2], chunk.shape[-1])
        positions = gathered.indices.clamp_min(0)
        if partition.position_index is not None:  # a region: rotate at full-protein positions
            positions = partition.position_index.to(positions.device)[positions]
        positions = positions.repeat(states.shape[0], 1)
        valid = gathered.mask.repeat(states.shape[0], 1)
        if mask is not None:
            runtime_mask = mask[:, gathered.indices.clamp_min(0)].bool()
            valid = valid & runtime_mask.reshape_as(valid)
        query_width = max(g.core_indices.numel() for g in partition.groups[start:stop])
        full_length = states.shape[1] if partition.full_length is None else partition.full_length
        updates = _pretrained_group_attention(
            attention, flat, valid, positions, full_length, query_width=query_width
        ).reshape(states.shape[0], stop - start, query_width, states.shape[-1])
        core_indices = torch.cat([g.core_indices for g in partition.groups[start:stop]]).to(states.device)
        core_updates = torch.cat([
            updates[:, i, :g.core_indices.numel()]
            for i, g in enumerate(partition.groups[start:stop])
        ], dim=1)
        # Match the former assignment's cast when autocast returns bf16
        # updates while the caller's residue states remain float32.
        assembled.index_copy_(1, core_indices, core_updates.to(assembled.dtype))
    return assembled
