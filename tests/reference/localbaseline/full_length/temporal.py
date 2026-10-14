"""Dense-reference and Hyena temporal mixing behind one trajectory contract."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..model.standalone_hyena import HyenaOperator


class TemporalMixer(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_frames: int,
        backend: str,
    ):
        super().__init__()
        if backend not in {"dense", "hyena"}:
            raise ValueError("backend must be 'dense' or 'hyena'")
        self.backend = backend
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        if backend == "dense":
            self.mixer = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        else:
            self.mixer = HyenaOperator(
                d_model=embed_dim,
                l_max=num_frames,
                order=2,
                filter_order=64,
            )

    def forward(
        self,
        states: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        sequence_batch_size: int | None = None,
        checkpoint_chunks: bool = False,
    ) -> torch.Tensor:
        """Mix [B,T,L,C] along T without changing caller-visible shape."""
        if states.ndim != 4 or states.shape[-1] != self.embed_dim:
            raise ValueError("states must have shape [B, T, L, C]")
        batch, frames, residues, channels = states.shape
        if frames > self.num_frames:
            raise ValueError("Input exceeds configured num_frames")
        sequences = states.transpose(1, 2).reshape(batch * residues, frames, channels)
        if mask is not None:
            if mask.shape != states.shape[:3]:
                raise ValueError("mask must have shape [B, T, L]")
            temporal_mask = mask.transpose(1, 2).reshape(batch * residues, frames).bool()
        else:
            temporal_mask = None

        if self.backend == "hyena":
            if temporal_mask is not None and not bool(temporal_mask.all()):
                raise ValueError("Hyena full-frame path requires an unpadded temporal axis")
        if sequence_batch_size is None:
            sequence_batch_size = sequences.shape[0]
        if sequence_batch_size <= 0:
            raise ValueError("sequence_batch_size must be positive")

        def mix(chunk: torch.Tensor, chunk_mask: torch.Tensor | None) -> torch.Tensor:
            if self.backend == "dense":
                output, _ = self.mixer(
                    chunk,
                    chunk,
                    chunk,
                    key_padding_mask=None if chunk_mask is None else ~chunk_mask,
                    need_weights=False,
                )
                return output
            return self.mixer(chunk)

        outputs = []
        for start in range(0, sequences.shape[0], sequence_batch_size):
            stop = min(sequences.shape[0], start + sequence_batch_size)
            chunk = sequences[start:stop]
            chunk_mask = None if temporal_mask is None else temporal_mask[start:stop]
            if checkpoint_chunks and chunk.requires_grad:
                outputs.append(
                    checkpoint(
                        lambda value, saved_mask=chunk_mask: mix(value, saved_mask),
                        chunk,
                        use_reentrant=False,
                    )
                )
            else:
                outputs.append(mix(chunk, chunk_mask))
        output = torch.cat(outputs, dim=0)
        return output.reshape(batch, residues, frames, channels).transpose(1, 2)


def _relative_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        (left - right).norm().div(left.norm().clamp_min(1e-8)).detach().item()
    )


def compare_temporal_mixers(
    dense: TemporalMixer,
    hyena: TemporalMixer,
    states: torch.Tensor,
    *,
    tolerance: float = 0.10,
    geometry_metric: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, float | bool]:
    """Measure independent dense/Hyena behavior on one deterministic fixture."""
    dense_input = states.detach().clone().requires_grad_(True)
    hyena_input = states.detach().clone().requires_grad_(True)

    start = perf_counter()
    dense_output = dense(dense_input)
    dense_seconds = perf_counter() - start
    dense_loss = dense_output.square().mean()
    dense_gradient = torch.autograd.grad(dense_loss, dense_input)[0]

    start = perf_counter()
    hyena_output = hyena(hyena_input)
    hyena_seconds = perf_counter() - start
    hyena_loss = hyena_output.square().mean()
    hyena_gradient = torch.autograd.grad(hyena_loss, hyena_input)[0]

    output_difference = _relative_difference(dense_output, hyena_output)
    loss_difference = float(
        (dense_loss - hyena_loss).abs().div(dense_loss.abs().clamp_min(1e-8)).item()
    )
    gradient_difference = _relative_difference(dense_gradient, hyena_gradient)
    if geometry_metric is None:
        geometry_difference = 0.0
    else:
        geometry_difference = _relative_difference(
            geometry_metric(dense_output), geometry_metric(hyena_output)
        )
    return {
        "relative_output_difference": output_difference,
        "relative_loss_difference": loss_difference,
        "relative_gradient_difference": gradient_difference,
        "relative_geometry_difference": geometry_difference,
        "dense_seconds": dense_seconds,
        "hyena_seconds": hyena_seconds,
        "within_tolerance": max(
            output_difference, loss_difference, gradient_difference, geometry_difference
        )
        <= tolerance,
    }
