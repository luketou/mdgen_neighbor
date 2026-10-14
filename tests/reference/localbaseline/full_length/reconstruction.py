"""Anchored, chain-aware reconstruction and bounded optional cleanup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class ReconstructionSegment:
    batch_index: int
    chain_id: int
    residue_indices: tuple[int, ...]


@dataclass(frozen=True)
class ReconstructionResult:
    raw: torch.Tensor
    reconstructed: torch.Tensor
    segments: tuple[ReconstructionSegment, ...]
    metrics: dict[str, object]


@dataclass(frozen=True)
class RelaxationResult:
    raw: torch.Tensor
    relaxed: torch.Tensor | None
    status: str
    displacement: dict[str, float]
    trajectory_change: float
    error: str | None = None


def _segments_for_batch(
    mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    segment_length: int,
    batch_index: int,
) -> list[ReconstructionSegment]:
    runs: list[list[int]] = []
    current: list[int] = []
    for index in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        contiguous = bool(current) and (
            int(chain_id[index]) == int(chain_id[current[-1]])
            and int(residue_index[index]) == int(residue_index[current[-1]]) + 1
        )
        if current and not contiguous:
            runs.append(current)
            current = []
        current.append(index)
    if current:
        runs.append(current)

    segments: list[ReconstructionSegment] = []
    for run in runs:
        for start in range(0, len(run), segment_length):
            members = tuple(run[start : start + segment_length])
            segments.append(
                ReconstructionSegment(
                    batch_index=batch_index,
                    chain_id=int(chain_id[members[0]]),
                    residue_indices=members,
                )
            )
    return segments


def _geometry_metrics(
    atom14: torch.Tensor,
    residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    *,
    boundary_pairs: set[tuple[int, int]] | None = None,
) -> dict[str, object]:
    _, frames, residues, _, _ = atom14.shape
    mask = residue_mask[:, None].expand(-1, frames, -1).bool()
    valid = mask[..., :-1] & mask[..., 1:]
    valid &= chain_id[:, None, 1:] == chain_id[:, None, :-1]
    valid &= residue_index[:, None, 1:] == residue_index[:, None, :-1] + 1
    if boundary_pairs is not None:
        boundary_mask = torch.zeros_like(valid)
        for batch_index, left_index in boundary_pairs:
            if left_index < residues - 1:
                boundary_mask[batch_index, :, left_index] = True
        valid &= boundary_mask

    cn = torch.linalg.vector_norm(
        atom14[..., :-1, 2, :] - atom14[..., 1:, 0, :], dim=-1
    )
    ca = torch.linalg.vector_norm(
        atom14[..., :-1, 1, :] - atom14[..., 1:, 1, :], dim=-1
    )
    cn_values = cn[valid]
    ca_values = ca[valid]
    denominator = max(1, cn_values.numel())
    bad_cn = ((cn_values < 1.1) | (cn_values > 1.6)).sum().item()
    bad_ca = ((ca_values < 3.4) | (ca_values > 4.2)).sum().item()

    def distribution(values: torch.Tensor) -> dict[str, float | int | None]:
        if values.numel() == 0:
            return {
                "count": 0,
                "min": None,
                "p05": None,
                "median": None,
                "p95": None,
                "max": None,
                "mean": None,
            }
        measured = values.detach().float()
        quantiles = torch.quantile(
            measured, torch.tensor([0.05, 0.5, 0.95], device=measured.device)
        )
        return {
            "count": int(measured.numel()),
            "min": float(measured.min().item()),
            "p05": float(quantiles[0].item()),
            "median": float(quantiles[1].item()),
            "p95": float(quantiles[2].item()),
            "max": float(measured.max().item()),
            "mean": float(measured.mean().item()),
        }

    return {
        "valid_pairs": int(cn_values.numel()),
        "bad_cn_ratio": float(bad_cn / denominator),
        "bad_ca_ratio": float(bad_ca / denominator),
        "cn_distribution": distribution(cn_values),
        "ca_distribution": distribution(ca_values),
    }


def reconstruct_anchored_segments(
    raw: torch.Tensor,
    *,
    residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    segment_length: int = 64,
) -> ReconstructionResult:
    """Rebuild local backbones, then close only adjacent same-chain segments."""
    if raw.ndim != 5 or raw.shape[-2:] != (14, 3):
        raise ValueError("raw must have shape [B, T, L, 14, 3]")
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    batch, _, residues = raw.shape[:3]
    expected = (batch, residues)
    for name, value in (
        ("residue_mask", residue_mask),
        ("chain_id", chain_id),
        ("residue_index", residue_index),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape [B, L]")

    segments: list[ReconstructionSegment] = []
    for batch_index in range(batch):
        segments.extend(
            _segments_for_batch(
                residue_mask[batch_index].bool(),
                chain_id[batch_index],
                residue_index[batch_index],
                segment_length,
                batch_index,
            )
        )

    reconstructed = raw.clone()
    anchor_displacements: list[torch.Tensor] = []
    boundary_pairs: set[tuple[int, int]] = set()
    by_chain: dict[tuple[int, int], list[ReconstructionSegment]] = {}
    for segment in segments:
        by_chain.setdefault((segment.batch_index, segment.chain_id), []).append(segment)

    for (batch_index, _), chain_segments in by_chain.items():
        chain_members = [i for segment in chain_segments for i in segment.residue_indices]
        first, last = chain_members[0], chain_members[-1]
        raw_direction = raw[batch_index, :, last, 1] - raw[batch_index, :, first, 1]
        norm = torch.linalg.vector_norm(raw_direction, dim=-1, keepdim=True)
        fallback = torch.zeros_like(raw_direction)
        fallback[..., 0] = 1.0
        direction = torch.where(norm > 1e-6, raw_direction / norm.clamp_min(1e-6), fallback)

        previous_last_ca: torch.Tensor | None = None
        previous_last_index: int | None = None
        for segment in chain_segments:
            members = list(segment.residue_indices)
            original_anchor = raw[batch_index, :, members[0], 1]
            if previous_last_ca is None:
                anchor = original_anchor
            else:
                anchor = previous_last_ca + 3.8 * direction
                boundary_pairs.add((batch_index, int(previous_last_index)))
            anchor_displacements.append(
                torch.linalg.vector_norm(anchor - original_anchor, dim=-1)
            )

            for offset, residue in enumerate(members):
                ca = anchor + (3.8 * offset) * direction
                translation = ca - raw[batch_index, :, residue, 1]
                reconstructed[batch_index, :, residue] += translation[:, None, :]
                reconstructed[batch_index, :, residue, 1] = ca
                reconstructed[batch_index, :, residue, 0] = ca - 1.236 * direction
                reconstructed[batch_index, :, residue, 2] = ca + 1.235 * direction
            previous_last_index = members[-1]
            previous_last_ca = reconstructed[batch_index, :, members[-1], 1]

    if anchor_displacements:
        anchor_rmsd = torch.cat(anchor_displacements).square().mean().sqrt().item()
    else:
        anchor_rmsd = 0.0
    metrics = {
        "before": _geometry_metrics(raw, residue_mask, chain_id, residue_index),
        "after": _geometry_metrics(
            reconstructed, residue_mask, chain_id, residue_index
        ),
        "boundary_before": _geometry_metrics(
            raw,
            residue_mask,
            chain_id,
            residue_index,
            boundary_pairs=boundary_pairs,
        ),
        "boundary_after": _geometry_metrics(
            reconstructed,
            residue_mask,
            chain_id,
            residue_index,
            boundary_pairs=boundary_pairs,
        ),
        "anchor_rmsd": float(anchor_rmsd),
        "segment_count": len(segments),
    }
    return ReconstructionResult(raw, reconstructed, tuple(segments), metrics)


def _trajectory_motion(coordinates: torch.Tensor) -> torch.Tensor:
    if coordinates.shape[1] < 2:
        return coordinates.new_tensor(0.0)
    ca = coordinates[..., 1, :]
    return torch.linalg.vector_norm(ca[:, 1:] - ca[:, :-1], dim=-1).mean()


def apply_optional_relaxation(
    raw: torch.Tensor,
    *,
    relaxer: Callable[[torch.Tensor], torch.Tensor] | None,
    max_rmsd: float = 0.5,
    max_trajectory_change: float = 0.1,
) -> RelaxationResult:
    """Run an injected lightweight relaxer and reject excessive cleanup."""
    if relaxer is None:
        return RelaxationResult(raw, None, "disabled", {"rmsd": 0.0, "max": 0.0}, 0.0)
    try:
        proposed = relaxer(raw.clone())
        if proposed.shape != raw.shape or not torch.isfinite(proposed).all():
            raise ValueError("Relaxer returned invalid coordinates")
        displacement = torch.linalg.vector_norm(proposed - raw, dim=-1)
        rmsd = displacement.square().mean().sqrt()
        raw_motion = _trajectory_motion(raw)
        proposed_motion = _trajectory_motion(proposed)
        trajectory_change = (
            (proposed_motion - raw_motion).abs() / raw_motion.clamp_min(1e-8)
        )
        accepted = rmsd <= max_rmsd and trajectory_change <= max_trajectory_change
        return RelaxationResult(
            raw=raw,
            relaxed=proposed if bool(accepted) else None,
            status="accepted" if bool(accepted) else "rejected",
            displacement={
                "rmsd": float(rmsd.item()),
                "max": float(displacement.max().item()),
            },
            trajectory_change=float(trajectory_change.item()),
        )
    except Exception as error:
        return RelaxationResult(
            raw, None, "failed", {"rmsd": 0.0, "max": 0.0}, 0.0, str(error)
        )


def make_openmm_backbone_relaxer(
    residue_mask: torch.Tensor,
    chain_id: torch.Tensor,
    residue_index: torch.Tensor,
    *,
    max_iterations: int = 50,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a restrained OpenMM minimizer for protein backbone atoms only."""

    def relax(coordinates: torch.Tensor) -> torch.Tensor:
        try:
            import openmm
            from openmm import unit
        except ImportError as error:
            raise RuntimeError("OpenMM is not installed; relaxation was not run") from error

        output = coordinates.detach().cpu().clone()
        batch, frames, residues = output.shape[:3]
        mask_cpu = residue_mask.detach().bool().cpu()
        chain_cpu = chain_id.detach().cpu()
        index_cpu = residue_index.detach().cpu()
        for batch_index in range(batch):
            valid = torch.nonzero(mask_cpu[batch_index], as_tuple=False).flatten().tolist()
            if not valid:
                continue
            particle_of: dict[tuple[int, int], int] = {}
            for ordinal, residue in enumerate(valid):
                for atom_slot in range(3):
                    particle_of[(residue, atom_slot)] = ordinal * 3 + atom_slot
            for frame_index in range(frames):
                system = openmm.System()
                for _ in range(3 * len(valid)):
                    system.addParticle(12.0)
                bonds = openmm.HarmonicBondForce()
                restraint = openmm.CustomExternalForce(
                    "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
                )
                restraint.addGlobalParameter("k", 500.0)
                for name in ("x0", "y0", "z0"):
                    restraint.addPerParticleParameter(name)
                positions = []
                for residue in valid:
                    backbone = output[batch_index, frame_index, residue, :3]
                    for atom_slot, position in enumerate(backbone):
                        positions.append(openmm.Vec3(*(position / 10.0).tolist()))
                        restraint.addParticle(
                            particle_of[(residue, atom_slot)], (position / 10.0).tolist()
                        )
                    bonds.addBond(
                        particle_of[(residue, 0)], particle_of[(residue, 1)], 0.1458, 30_000.0
                    )
                    bonds.addBond(
                        particle_of[(residue, 1)], particle_of[(residue, 2)], 0.1525, 30_000.0
                    )
                for left, right in zip(valid[:-1], valid[1:]):
                    if (
                        int(chain_cpu[batch_index, left]) == int(chain_cpu[batch_index, right])
                        and int(index_cpu[batch_index, right])
                        == int(index_cpu[batch_index, left]) + 1
                    ):
                        bonds.addBond(
                            particle_of[(left, 2)], particle_of[(right, 0)], 0.1329, 30_000.0
                        )
                system.addForce(bonds)
                system.addForce(restraint)
                integrator = openmm.VerletIntegrator(0.001)
                context = openmm.Context(system, integrator)
                context.setPositions(positions * unit.nanometer)
                openmm.LocalEnergyMinimizer.minimize(
                    context, maxIterations=max_iterations
                )
                relaxed = context.getState(getPositions=True).getPositions(asNumpy=True)
                relaxed = torch.tensor(
                    relaxed.value_in_unit(unit.angstrom), dtype=output.dtype
                ).reshape(len(valid), 3, 3)
                for ordinal, residue in enumerate(valid):
                    output[batch_index, frame_index, residue, :3] = relaxed[ordinal]
                del context, integrator
        return output.to(device=coordinates.device, dtype=coordinates.dtype)

    return relax
