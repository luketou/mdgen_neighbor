#!/usr/bin/env python3
"""Measure FP32 single-forward peak memory across residue lengths."""

# a4-l-scaling:start
import argparse
import gc
import json
from pathlib import Path

import torch

from mdgen.rigid_utils import Rigid
from mdgen.wrapper import NewMDGenWrapper


DEFAULT_LENGTHS = (256, 1000, 2500, 5000, 7500)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark one full MDGen model forward at each residue length."
    )
    parser.add_argument("--sim_ckpt", required=True, help="Path to atlas.ckpt.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path.")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=DEFAULT_LENGTHS,
        help="Residue lengths to measure.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=250,
        help="Trajectory frames T; Atlas inference uses 250.",
    )
    parser.add_argument("--manual_attention", action="store_true")  # sdpa-route
    # sdpa-diagnostics:start
    parser.add_argument(
        "--print_sdpa_backend",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Profile one forward at the first requested length and write the "
            "actual PyTorch SDPA backend report."
        ),
    )
    # sdpa-diagnostics:end
    parser.add_argument("--seed", type=int, default=None)  # seed-deterministic-args
    # seed-deterministic-args:start
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use deterministic algorithms when explicitly requested.",
    )
    # seed-deterministic-args:end
    parser.add_argument(
        "--continue_after_oom",
        action="store_true",
        help="Continue to larger lengths after an OOM (normally unnecessary).",
    )
    return parser.parse_args()


def write_report(path, report):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


def attention_paths(model):
    paths = {}
    for name, module in model.named_modules():
        if hasattr(module, "last_attention_backend"):
            paths[name] = {
                "last_attention_backend": module.last_attention_backend,
                "last_sdpa_fallback_reason": module.last_sdpa_fallback_reason,
            }
    return paths


# sdpa-diagnostics:start
def get_sdpa_backend(operator_name):
    backend_operators = {
        "_scaled_dot_product_flash_attention": "FLASH_ATTENTION",
        "_scaled_dot_product_efficient_attention": "EFFICIENT_ATTENTION",
        "_scaled_dot_product_cudnn_attention": "CUDNN_ATTENTION",
        "_scaled_dot_product_attention_math": "MATH",
    }
    for operator, backend in backend_operators.items():
        if operator in operator_name:
            return backend
    return None


def write_sdpa_backend_report(
    path, model, profiler, length, num_frames, use_sdpa, device
):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "SDPA backend report",
        "scope: A4 single LatentMDGenModel forward",
        f'requested_attention_path: {"sdpa" if use_sdpa else "manual"}',
        f"batch_size: 1",
        f"num_frames: {num_frames}",
        f"residue_length: {length}",
        "dtype: float32",
        f"torch_version: {torch.__version__}",
        f"cuda_wheel: {torch.version.cuda}",
        f"gpu: {torch.cuda.get_device_name(device)}",
        "",
        "Enabled PyTorch SDPA backends:",
    ]
    enabled_checks = (
        ("FLASH_ATTENTION", "flash_sdp_enabled"),
        ("EFFICIENT_ATTENTION", "mem_efficient_sdp_enabled"),
        ("CUDNN_ATTENTION", "cudnn_sdp_enabled"),
        ("MATH", "math_sdp_enabled"),
    )
    for backend, check_name in enabled_checks:
        check = getattr(torch.backends.cuda, check_name, None)
        lines.append(f'{backend}: {check() if check is not None else "unknown"}')

    lines.extend(["", "MDGen attention modules:"])
    modules = attention_paths(model.model)
    if not modules:
        lines.append("No attention modules expose backend information.")
    for name, details in modules.items():
        backend_path = details.get(
            "last_attention_backend", details.get("path")
        )
        fallback_reason = details.get(
            "last_sdpa_fallback_reason", details.get("fallback_reason")
        )
        message = f'{name}: path={backend_path or "not_executed"}'
        if fallback_reason is not None:
            message += f", fallback={fallback_reason}"
        lines.append(message)

    lines.extend(["", "PyTorch SDPA operators observed:"])
    observed_backends = set()
    observed_operators = []
    for event in profiler.key_averages(group_by_input_shape=True):
        if "scaled_dot_product" not in event.key:
            continue
        backend = get_sdpa_backend(event.key)
        if backend is not None:
            observed_backends.add(backend)
        observed_operators.append((event, backend))

    if observed_backends:
        lines.append(f'selected_backends: {", ".join(sorted(observed_backends))}')
    else:
        lines.append("selected_backends: none detected")
    if not observed_operators:
        lines.append("No scaled-dot-product attention operators were observed.")
    for event, backend in observed_operators:
        self_device_time = getattr(
            event,
            "self_device_time_total",
            getattr(event, "self_cuda_time_total", 0.0),
        )
        device_time = getattr(
            event,
            "device_time_total",
            getattr(event, "cuda_time_total", 0.0),
        )
        lines.extend(
            [
                f"operator: {event.key}",
                f'  backend: {backend or "dispatcher/unknown"}',
                f"  calls: {event.count}",
                f"  input_shapes: {event.input_shapes}",
                f"  self_cpu_time_total_us: {event.self_cpu_time_total}",
                f"  cpu_time_total_us: {event.cpu_time_total}",
                f"  self_device_time_total_us: {self_device_time}",
                f"  device_time_total_us: {device_time}",
            ]
        )

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"SDPA backend report saved to: {path}")
# sdpa-diagnostics:end


def make_forward_inputs(model, length, num_frames, device):
    batch_size = 1
    latent_dim = model.latent_dim
    x = torch.zeros(
        batch_size,
        num_frames,
        length,
        latent_dim,
        dtype=torch.float32,
        device=device,
    )
    mask = torch.ones(
        batch_size, num_frames, length, dtype=torch.float32, device=device
    )
    x_cond = torch.zeros_like(x)
    x_cond_mask = torch.zeros(
        batch_size, num_frames, length, dtype=torch.long, device=device
    )
    if model.args.sim_condition:
        x_cond_mask[:, 0] = 1

    # position-id-model-routing: r2 intentionally leaves position_ids unset.
    return {
        "x": x,
        "t": torch.full((batch_size,), 0.5, dtype=torch.float32, device=device),
        "mask": mask,
        "start_frames": Rigid.identity(
            (batch_size, length),
            dtype=torch.float32,
            device=device,
            requires_grad=False,
            fmt="rot_mat",
        ),
        "end_frames": None,
        "x_cond": x_cond,
        "x_cond_mask": x_cond_mask,
        "aatype": torch.zeros(
            batch_size, length, dtype=torch.long, device=device
        ),
    }


# sdpa-diagnostics:start
def profile_sdpa_backend(
    model, length, num_frames, device, output_path, use_sdpa
):
    inputs = None
    output = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    try:
        inputs = make_forward_inputs(model, length, num_frames, device)
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(
            activities=activities, record_shapes=True
        ) as profiler:
            with torch.inference_mode():
                output = model.model(**inputs)
        torch.cuda.synchronize()
        write_sdpa_backend_report(
            output_path,
            model,
            profiler,
            length,
            num_frames,
            use_sdpa,
            device,
        )
    finally:
        del output
        del inputs
        gc.collect()
        torch.cuda.empty_cache()
# sdpa-diagnostics:end


def memory_stats():
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    return {
        "peak_memory_allocated_bytes": allocated,
        "peak_memory_allocated_gib": allocated / 1024**3,
        "peak_memory_reserved_bytes": reserved,
        "peak_memory_reserved_gib": reserved / 1024**3,
    }


def run_case(model, length, num_frames, device):
    inputs = None
    output = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()

    try:
        inputs = make_forward_inputs(model, length, num_frames, device)
        with torch.inference_mode():
            output = model.model(**inputs)
        torch.cuda.synchronize()
        return {
            "length": length,
            "oom": False,
            "baseline_memory_allocated_bytes": baseline_allocated,
            "baseline_memory_allocated_gib": baseline_allocated / 1024**3,
            **memory_stats(),
            "output_shape": list(output.shape),
            "attention_modules": attention_paths(model.model),
        }
    except RuntimeError as error:
        if not isinstance(error, torch.cuda.OutOfMemoryError) and "out of memory" not in str(error).lower():
            raise
        return {
            "length": length,
            "oom": True,
            "peak_is_pre_oom_lower_bound": True,
            "baseline_memory_allocated_bytes": baseline_allocated,
            "baseline_memory_allocated_gib": baseline_allocated / 1024**3,
            **memory_stats(),
            "error": str(error),
        }
    finally:
        del output
        del inputs
        gc.collect()
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    use_sdpa = not args.manual_attention  # sdpa-route
    if not torch.cuda.is_available():
        raise RuntimeError("A4 L-scaling requires a CUDA GPU")
    if args.num_frames <= 0 or any(length <= 0 for length in args.lengths):
        raise ValueError("--num_frames and every --lengths value must be positive")

    # seed-initialization:start
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    # seed-initialization:end
    torch.use_deterministic_algorithms(args.deterministic)  # deterministic-execution
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda", torch.cuda.current_device())
    model = NewMDGenWrapper.load_from_checkpoint(
        args.sim_ckpt,
        use_sdpa=use_sdpa,  # sdpa-route
        weights_only=False,
    )
    model.eval().float().to(device)
    model.args.grad_checkpointing = False

    # sdpa-diagnostics:start
    if args.print_sdpa_backend is not None:
        profile_sdpa_backend(
            model,
            args.lengths[0],
            args.num_frames,
            device,
            args.print_sdpa_backend,
            use_sdpa,  # sdpa-route
        )
    # sdpa-diagnostics:end

    report = {
        "benchmark": "A4 L-scaling single model forward",
        "checkpoint": str(Path(args.sim_ckpt).resolve()),
        "measurement_scope": (
            "Full LatentMDGenModel forward in torch.inference_mode; peak includes "
            "model parameters, synthetic inputs, and forward temporaries, but excludes "
            "checkpoint-loading peak."
        ),
        "attention_path": "sdpa" if use_sdpa else "manual",  # sdpa-route
        "dtype": "float32",
        "batch_size": 1,
        "num_frames": args.num_frames,
        "grad_checkpointing": False,
        "deterministic": args.deterministic,  # deterministic-execution
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": False,
        "seed": args.seed,
        "torch_version": torch.__version__,
        "cuda_wheel": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "lengths_requested": args.lengths,
        "results": [],
    }
    write_report(args.output, report)

    for length in args.lengths:
        print(f"A4: attention={report['attention_path']} B=1 T={args.num_frames} L={length}")
        result = run_case(model, length, args.num_frames, device)
        report["results"].append(result)
        write_report(args.output, report)
        status = "OOM" if result["oom"] else "success"
        print(
            f"A4: L={length} {status}, "
            f"peak_allocated={result['peak_memory_allocated_gib']:.3f} GiB"
        )
        if result["oom"] and not args.continue_after_oom:
            break

    print(f"A4 report saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
# a4-l-scaling:end
