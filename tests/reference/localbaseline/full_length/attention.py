"""Replaceable dense-in-group attention with core-only assembly."""

from __future__ import annotations

from dataclasses import replace
import resource
from time import perf_counter

import torch
from torch import nn
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


def run_pretrained_sdpa_attention(module, states, mask=None):
    """Full self-attention using the existing weights and sequence RoPE slots.

    States are [N, L, C]; a true mask entry denotes a valid key. Temporal
    attention uses this path without spatial gathering or new parameters.
    """
    if states.ndim != 3 or (mask is not None and mask.shape != states.shape[:2]):
        raise ValueError("states must be [N, L, C] and mask must be [N, L]")
    attention = module.attn
    if not attention.qkv_same_dim or attention.onnx_trace:
        raise ValueError("Pretrained SDPA requires same-dimension self-attention")
    valid = torch.ones(states.shape[:2], dtype=torch.bool, device=states.device) if mask is None else mask.bool()
    positions = torch.arange(states.shape[1], device=states.device)[None].expand(states.shape[0], -1)
    return _pretrained_group_attention(attention, states, valid, positions, states.shape[1])


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
    pairwise biases or cross-attention. The standalone fixture below remains
    available for backend experiments, but is not used by the trained model.
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
        positions = gathered.indices.clamp_min(0).repeat(states.shape[0], 1)
        valid = gathered.mask.repeat(states.shape[0], 1)
        if mask is not None:
            runtime_mask = mask[:, gathered.indices.clamp_min(0)].bool()
            valid = valid & runtime_mask.reshape_as(valid)
        query_width = max(g.core_indices.numel() for g in partition.groups[start:stop])
        updates = _pretrained_group_attention(
            attention, flat, valid, positions, states.shape[1], query_width=query_width
        ).reshape(states.shape[0], stop - start, query_width, states.shape[-1])
        core_indices = torch.cat([g.core_indices for g in partition.groups[start:stop]])
        core_updates = torch.cat([
            updates[:, i, :g.core_indices.numel()]
            for i, g in enumerate(partition.groups[start:stop])
        ], dim=1)
        # Match the former assignment's cast when autocast returns bf16
        # updates while the caller's residue states remain float32.
        assembled.index_copy_(1, core_indices, core_updates.to(assembled.dtype))
    return assembled


class LocalSelfAttention(nn.Module):
    """Self-attention whose public contract is independent of its backend."""

    SUPPORTED_BACKENDS = frozenset({"sdpa", "flex"})

    def __init__(self, embed_dim: int, num_heads: int, backend: str = "sdpa"):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported attention backend: {backend}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.backend = backend
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.output = nn.Linear(embed_dim, embed_dim)

    def _project(self, states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch, length, _ = states.shape
        qkv = self.qkv(states).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        return tuple(value.transpose(1, 2) for value in (q, k, v))

    def _sdpa(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        attention_mask = None
        if mask is not None:
            attention_mask = mask[:, None, None, :]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)

    def _flex(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        try:
            from torch.nn.attention.flex_attention import flex_attention
        except ImportError as error:
            raise RuntimeError("FlexAttention is unavailable in this PyTorch build") from error
        if mask is None or bool(mask.all()):
            return flex_attention(q, k, v)

        def score_mod(score, batch, head, query, key):
            del head, query
            return torch.where(mask[batch, key], score, -torch.inf)

        return flex_attention(q, k, v, score_mod=score_mod)

    def _attend(self, states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        q, k, v = self._project(states)
        if self.backend == "sdpa":
            attended = self._sdpa(q, k, v, mask)
        else:
            attended = self._flex(q, k, v, mask)
        attended = attended.transpose(1, 2).reshape(states.shape)
        output = self.output(attended)
        if mask is not None:
            output = output * mask.unsqueeze(-1)
        return output

    def forward_dense(
        self, states: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if states.ndim != 3 or states.shape[-1] != self.embed_dim:
            raise ValueError("states must have shape [N, L, C]")
        return self._attend(states, mask)

    def forward(
        self,
        states: torch.Tensor,
        partition: SpatialPartition,
        *,
        group_batch_size: int = 1,
    ) -> torch.Tensor:
        """Attend inside groups and assemble only their core positions."""
        if states.ndim != 3 or states.shape[1] != partition.residue_mask.numel():
            raise ValueError("states must have shape [N, L, C] for this partition")
        if group_batch_size <= 0:
            raise ValueError("group_batch_size must be positive")
        if not partition.groups:
            return torch.zeros_like(states)

        gathered = partition.gather(states.transpose(0, 1))
        # partition.gather returns [G, Q, N, C].
        grouped = gathered.values.permute(2, 0, 1, 3)
        outputs = torch.zeros_like(grouped)
        group_count = grouped.shape[1]
        for start in range(0, group_count, group_batch_size):
            stop = min(group_count, start + group_batch_size)
            chunk = grouped[:, start:stop]
            flat = chunk.reshape(-1, chunk.shape[-2], chunk.shape[-1])
            flat_mask = gathered.mask[start:stop].repeat(states.shape[0], 1)
            outputs[:, start:stop] = self._attend(flat, flat_mask).reshape_as(chunk)

        local_outputs = outputs.permute(1, 2, 0, 3)
        assembled = partition.scatter_core(local_outputs, gathered)
        return assembled.permute(1, 0, 2)


def compare_attention_backends(
    reference: LocalSelfAttention,
    candidate: LocalSelfAttention,
    states: torch.Tensor,
    partition: SpatialPartition,
) -> dict[str, float | int | bool]:
    """Compare outputs, input gradients, runtime, and peak memory."""
    candidate.load_state_dict(reference.state_dict())
    group_batch_size = max(1, len(partition.groups))

    def measure(module: LocalSelfAttention):
        value = states.detach().clone().requires_grad_(True)
        if value.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(value.device)
        start = perf_counter()
        output = module(value, partition, group_batch_size=group_batch_size)
        gradient = torch.autograd.grad(output.square().mean(), value)[0]
        elapsed = perf_counter() - start
        if value.device.type == "cuda":
            peak_memory = int(torch.cuda.max_memory_allocated(value.device))
        else:
            peak_memory = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        return output.detach(), gradient.detach(), elapsed, peak_memory

    reference_output, reference_gradient, reference_seconds, reference_memory = measure(
        reference
    )
    candidate_output, candidate_gradient, candidate_seconds, candidate_memory = measure(
        candidate
    )
    difference = (reference_output - candidate_output).abs()
    gradient_difference = (reference_gradient - candidate_gradient).abs()
    equivalent = bool(
        torch.allclose(reference_output, candidate_output, atol=1e-5, rtol=1e-4)
        and torch.allclose(
            reference_gradient, candidate_gradient, atol=1e-5, rtol=1e-4
        )
    )
    return {
        "max_output_difference": float(difference.max().item()),
        "mean_output_difference": float(difference.mean().item()),
        "max_gradient_difference": float(gradient_difference.max().item()),
        "mean_gradient_difference": float(gradient_difference.mean().item()),
        "reference_seconds": reference_seconds,
        "candidate_seconds": candidate_seconds,
        "reference_peak_memory_bytes": reference_memory,
        "candidate_peak_memory_bytes": candidate_memory,
        "finite": bool(torch.isfinite(candidate_output).all().item()),
        "equivalent": equivalent,
        "candidate_improves_measured_performance": bool(
            candidate_seconds < reference_seconds or candidate_memory < reference_memory
        ),
        "candidate_preferred": bool(
            equivalent
            and (candidate_seconds < reference_seconds or candidate_memory < reference_memory)
        ),
    }
