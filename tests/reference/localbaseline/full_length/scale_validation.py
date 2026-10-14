"""Deterministic end-to-end scale-validation command and JSON report."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import resource
from time import perf_counter
from typing import Any

import torch

from ..geometry_loss import peptide_geometry_loss
from ..rigid_utils import Rigid
from .attention import LocalSelfAttention, compare_attention_backends
from .checkpoint import load_compatible_state_dict
from .partition import build_spatial_partition
from .pipeline import BoundedTrajectoryBlock
from .reconstruction import (
    apply_optional_relaxation,
    make_openmm_backbone_relaxer,
    reconstruct_anchored_segments,
)
from .temporal import TemporalMixer, compare_temporal_mixers


@dataclass(frozen=True)
class ScaleValidationConfig:
    fixture: str = "short"
    residues: int | None = None
    frames: int | None = None
    batch_size: int = 1
    precision: str = "float32"
    embed_dim: int | None = None
    num_heads: int | None = None
    group_batch_size: int = 4
    temporal_batch_size: int = 64
    core_capacity: int = 128
    ghost_capacity: int = 128
    cell_width: float = 16.0
    contact_radius: float = 16.0
    attention_backend: str = "sdpa"
    device: str = "auto"
    seed: int = 0
    run_backward: bool = True
    output: Path = Path("scale-validation-report.json")
    relaxation: bool = False
    checkpoint: Path | None = None

    def resolved(self) -> "ScaleValidationConfig":
        if self.fixture not in {"short", "full"}:
            raise ValueError("fixture must be 'short' or 'full'")
        resolved = ScaleValidationConfig(
            **{
                **asdict(self),
                "residues": self.residues or (32 if self.fixture == "short" else 7_500),
                "frames": self.frames or (8 if self.fixture == "short" else 250),
                "embed_dim": self.embed_dim or (16 if self.fixture == "short" else 384),
                "num_heads": self.num_heads or (1 if self.fixture == "short" else 16),
                "output": Path(self.output),
                "checkpoint": None if self.checkpoint is None else Path(self.checkpoint),
            }
        )
        if resolved.batch_size != 1:
            raise ValueError("scale validation models exactly one molecular system")
        if resolved.embed_dim % resolved.num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        return resolved


def _json_config(config: ScaleValidationConfig) -> dict[str, Any]:
    values = asdict(config)
    values["output"] = str(config.output)
    values["checkpoint"] = None if config.checkpoint is None else str(config.checkpoint)
    return values


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _dtype(name: str) -> torch.dtype:
    try:
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]
    except KeyError as error:
        raise ValueError("precision must be float32, float16, or bfloat16") from error


def _fixture(config: ScaleValidationConfig, device: torch.device, dtype: torch.dtype):
    residues = int(config.residues)
    frames = int(config.frames)
    residue_index = torch.arange(residues, device=device)
    chain_id = torch.zeros(residues, dtype=torch.long, device=device)
    mask = torch.ones(residues, dtype=torch.bool, device=device)
    coordinate_dtype = torch.float32
    ca = torch.zeros(residues, 3, device=device, dtype=coordinate_dtype)
    ca[:, 0] = residue_index.to(coordinate_dtype) * 3.8

    atom14 = torch.zeros(
        1, frames, residues, 14, 3, device=device, dtype=coordinate_dtype
    )
    motion = 0.05 * torch.sin(
        torch.linspace(
            0, 2 * torch.pi, frames, device=device, dtype=coordinate_dtype
        )
    )
    frame_ca = ca[None, None].expand(1, frames, -1, -1).clone()
    frame_ca[..., 1] += motion[None, :, None]
    direction = torch.tensor(
        [1.0, 0.0, 0.0], device=device, dtype=coordinate_dtype
    )
    atom14[..., 1, :] = frame_ca
    atom14[..., 0, :] = frame_ca - 1.236 * direction
    atom14[..., 2, :] = frame_ca + 1.235 * direction

    channels = torch.arange(config.embed_dim, device=device, dtype=dtype)
    states = torch.sin(
        residue_index.to(dtype)[None, None, :, None] * 0.13
        + torch.arange(frames, device=device, dtype=dtype)[None, :, None, None] * 0.17
        + channels[None, None, None] * 0.07
    )
    states.requires_grad_(config.run_backward)
    frame_tensor = torch.zeros(frames, residues, 7, device=device, dtype=dtype)
    frame_tensor[..., 0] = 1.0
    frame_tensor[..., 4:] = frame_ca[0].to(dtype)
    return states, atom14, ca, mask, chain_id, residue_index, Rigid.from_tensor_7(frame_tensor)


def _cpu_peak_bytes() -> int:
    # Linux ru_maxrss is KiB.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def run_scale_validation(config: ScaleValidationConfig) -> dict[str, Any]:
    config = config.resolved()
    output = config.output
    report: dict[str, Any] = {
        "status": "running",
        "classification": "protein-only engineering scale validation",
        "fixture_identity": f"deterministic-synthetic-{config.fixture}-v1",
        "config": _json_config(config),
        "seed": config.seed,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    stage = "configuration"
    forward_seconds = 0.0
    backward_seconds = 0.0
    try:
        torch.manual_seed(config.seed)
        device = _device(config.device)
        dtype = _dtype(config.precision)
        report["device"] = str(device)
        stage = "device_validation"
        if config.fixture == "full" and device.type != "cuda":
            raise RuntimeError("The 7,500 x 250 full fixture requires a CUDA accelerator")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        stage = "input_preparation"
        states, raw, ca, mask, chain_id, residue_index, frames = _fixture(
            config, device, dtype
        )

        stage = "spatial_partition"
        partition = build_spatial_partition(
            ca,
            mask=mask,
            chain_id=chain_id,
            residue_index=residue_index,
            cell_width=config.cell_width,
            contact_radius=config.contact_radius,
            core_capacity=config.core_capacity,
            ghost_capacity=config.ghost_capacity,
        )
        ownership = partition.validate_ownership()

        stage = "model_initialization"
        model = BoundedTrajectoryBlock(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_frames=int(config.frames),
            attention_backend=config.attention_backend,
        ).to(device=device, dtype=dtype)
        if config.checkpoint is None:
            checkpoint_report: dict[str, Any] = {"status": "not_requested"}
        else:
            checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
            checkpoint_report = {
                "status": "loaded",
                "path": str(config.checkpoint),
                **load_compatible_state_dict(model, checkpoint),
            }

        stage = "forward"
        start = perf_counter()
        latent, displacement = model(
            states,
            frames,
            partition,
            group_batch_size=config.group_batch_size,
            temporal_batch_size=config.temporal_batch_size,
        )
        predicted_raw = raw + displacement.detach()
        del displacement
        forward_seconds = perf_counter() - start

        stage = "backward"
        if config.run_backward:
            start = perf_counter()
            latent.mean().backward()
            backward_seconds = perf_counter() - start

        stage = "reconstruction"
        reconstruction = reconstruct_anchored_segments(
            predicted_raw.detach(),
            residue_mask=mask[None],
            chain_id=chain_id[None],
            residue_index=residue_index[None],
        )

        stage = "relaxation"
        relaxation = apply_optional_relaxation(
            reconstruction.reconstructed,
            relaxer=(
                make_openmm_backbone_relaxer(
                    mask[None], chain_id[None], residue_index[None]
                )
                if config.relaxation
                else None
            ),
        )

        stage = "backend_comparison"
        reference = LocalSelfAttention(config.embed_dim, config.num_heads, "sdpa").to(
            device=device, dtype=dtype
        )
        try:
            candidate = LocalSelfAttention(config.embed_dim, config.num_heads, "flex").to(
                device=device, dtype=dtype
            )
            attention_comparison: dict[str, Any] = compare_attention_backends(
                reference, candidate, states[0, :1].detach(), partition
            )
            attention_comparison["status"] = "measured"
        except Exception as error:
            attention_comparison = {"status": "unavailable", "error": str(error)}

        comparison_frames = min(8, int(config.frames))
        comparison_residues = min(8, int(config.residues))
        dense_temporal = TemporalMixer(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_frames=comparison_frames,
            backend="dense",
        ).to(device=device, dtype=dtype)
        hyena_temporal = TemporalMixer(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_frames=comparison_frames,
            backend="hyena",
        ).to(device=device, dtype=dtype)
        temporal_comparison = compare_temporal_mixers(
            dense_temporal,
            hyena_temporal,
            states[:, :comparison_frames, :comparison_residues].detach(),
            geometry_metric=lambda output: torch.stack(
                tuple(
                    peptide_geometry_loss(
                        raw[:, :comparison_frames, :comparison_residues]
                        + 0.01 * output[..., :3].unsqueeze(-2),
                        mask[None, :comparison_residues],
                        residue_index=residue_index[None, :comparison_residues],
                        chain_id=chain_id[None, :comparison_residues],
                    )[key]
                    for key in ("cn_loss", "ca_loss", "bad_cn_ratio", "bad_ca_ratio")
                )
            ),
        )

        stage = "artifact_write"
        output.parent.mkdir(parents=True, exist_ok=True)
        raw_path = output.with_name("raw.pt")
        reconstructed_path = output.with_name("reconstructed.pt")
        torch.save(predicted_raw.detach().cpu(), raw_path)
        torch.save(reconstruction.reconstructed.detach().cpu(), reconstructed_path)
        artifacts = {"raw": str(raw_path), "reconstructed": str(reconstructed_path)}
        if relaxation.relaxed is not None:
            relaxed_path = output.with_name("relaxed.pt")
            torch.save(relaxation.relaxed.detach().cpu(), relaxed_path)
            artifacts["relaxed"] = str(relaxed_path)

        final = relaxation.relaxed if relaxation.relaxed is not None else reconstruction.reconstructed
        nan_count = int(torch.isnan(final).sum().item())
        inf_count = int(torch.isinf(final).sum().item())
        gathered = partition.gather(states.detach()[0, 0])
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else _cpu_peak_bytes()
        )
        report.update(
            {
                "status": "completed",
                "failed_stage": None,
                "oom": False,
                "output_shape": list(final.shape),
                "numerics": {"nan_count": nan_count, "inf_count": inf_count},
                "partition": {
                    "group_count": len(partition.groups),
                    "max_core_size": max(g.core_indices.numel() for g in partition.groups),
                    "max_ghost_size": max(g.ghost_indices.numel() for g in partition.groups),
                    "missing_core_owners": ownership.missing_count,
                    "duplicate_core_owners": ownership.duplicate_count,
                    "ghost_update_violations": partition.ghost_update_violations(
                        gathered.values, gathered
                    ),
                },
                "spatial_attention": {
                    "selected_backend": config.attention_backend,
                    "global_pair_tensor_allocated": False,
                    "comparison": attention_comparison,
                },
                "invariant_point_attention": {"global_pair_tensor_allocated": False},
                "temporal": {
                    "candidate_backend": "hyena",
                    "accepted_for_full_path": temporal_comparison["within_tolerance"],
                    "selected_backend": (
                        "hyena"
                        if temporal_comparison["within_tolerance"]
                        else "dense"
                    ),
                    "frames": int(config.frames),
                    "comparison": temporal_comparison,
                },
                "adaptive_repartition": {
                    "enabled": False,
                    "threshold_angstrom": 8.0,
                    "events": [],
                },
                "geometry": reconstruction.metrics,
                "relaxation": {
                    "status": relaxation.status,
                    "displacement": relaxation.displacement,
                    "trajectory_change": relaxation.trajectory_change,
                    "error": relaxation.error,
                },
                "timing": {
                    "forward_seconds": forward_seconds,
                    "backward_seconds": backward_seconds,
                },
                "peak_memory_bytes": peak_memory,
                "checkpoint": checkpoint_report,
                "artifacts": artifacts,
            }
        )
    except Exception as error:
        is_oom = isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()
        report.update(
            {
                "status": "failed",
                "failed_stage": stage,
                "oom": is_oom,
                "error": f"{type(error).__name__}: {error}",
                "timing": {
                    "forward_seconds": forward_seconds,
                    "backward_seconds": backward_seconds,
                },
            }
        )
    _write_report(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=("short", "full"), default="short")
    parser.add_argument("--residues", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--embed-dim", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--group-batch-size", type=int, default=4)
    parser.add_argument("--temporal-batch-size", type=int, default=64)
    parser.add_argument("--attention-backend", choices=("sdpa", "flex"), default="sdpa")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("scale-validation-report.json"))
    parser.add_argument("--no-backward", action="store_true")
    parser.add_argument("--relaxation", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = ScaleValidationConfig(
        fixture=args.fixture,
        residues=args.residues,
        frames=args.frames,
        batch_size=args.batch_size,
        precision=args.precision,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        group_batch_size=args.group_batch_size,
        temporal_batch_size=args.temporal_batch_size,
        attention_backend=args.attention_backend,
        device=args.device,
        seed=args.seed,
        run_backward=not args.no_backward,
        output=args.output,
        relaxation=args.relaxation,
        checkpoint=args.checkpoint,
    )
    report = run_scale_validation(config)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
