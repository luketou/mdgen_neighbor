"""Differentiable peptide-chain geometry losses and metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _expand_residue_field(field: torch.Tensor, frames: int) -> torch.Tensor:
    if field.ndim == 2:
        return field[:, None, :].expand(-1, frames, -1)
    if field.ndim == 3:
        return field
    raise ValueError(f"Expected [B, L] or [B, T, L], got {tuple(field.shape)}")


def peptide_geometry_loss(
    atom14: torch.Tensor,
    residue_mask: torch.Tensor,
    residue_index: torch.Tensor | None = None,
    chain_id: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Measure peptide continuity for Atom14 coordinates shaped [B,T,L,14,3]."""
    if atom14.ndim != 5 or atom14.shape[-2:] != (14, 3):
        raise ValueError(f"Expected atom14 [B,T,L,14,3], got {tuple(atom14.shape)}")

    _, frames, residues, _, _ = atom14.shape
    mask = _expand_residue_field(residue_mask, frames).bool()
    pair_mask = mask[..., :-1] & mask[..., 1:]

    if residue_index is not None:
        indices = _expand_residue_field(residue_index, frames)
        pair_mask &= indices[..., 1:] == indices[..., :-1] + 1
    if chain_id is not None:
        chains = _expand_residue_field(chain_id, frames)
        pair_mask &= chains[..., 1:] == chains[..., :-1]

    # Atom14 backbone ordering is N=0, CA=1, C=2.
    cn_distance = torch.linalg.vector_norm(
        atom14[..., :-1, 2, :] - atom14[..., 1:, 0, :], dim=-1
    )
    ca_distance = torch.linalg.vector_norm(
        atom14[..., :-1, 1, :] - atom14[..., 1:, 1, :], dim=-1
    )

    weights = pair_mask.to(atom14.dtype)
    denominator = weights.sum().clamp_min(1.0)
    cn_loss = (
        F.smooth_l1_loss(cn_distance, torch.full_like(cn_distance, 1.329), reduction="none")
        * weights
    ).sum() / denominator
    ca_loss = (
        F.smooth_l1_loss(ca_distance, torch.full_like(ca_distance, 3.8), reduction="none")
        * weights
    ).sum() / denominator

    bad_cn = ((cn_distance < 1.1) | (cn_distance > 1.6)) & pair_mask
    bad_ca = ((ca_distance < 3.4) | (ca_distance > 4.2)) & pair_mask
    return {
        "cn_loss": cn_loss,
        "ca_loss": ca_loss,
        "bad_cn_ratio": bad_cn.to(atom14.dtype).sum() / denominator,
        "bad_ca_ratio": bad_ca.to(atom14.dtype).sum() / denominator,
        "valid_pairs": pair_mask.sum(),
    }
