"""Full-length FP32 generation profile: NFE, wall clock and kernel launches (#9 Ticket 8).

Usage:
    python -m mdgen.full_length.generation_profile --sim_ckpt atlas.ckpt --output report.json \
        (--synthetic_residues 7500 | --pdb_id NAME --data_dir D --split S [--suffix _R1]) \
        [--rollouts 1] [--seed 0] [--ode_atol 1e-6 --ode_rtol 1e-3] \
        [--spatial_cell_width ... other local overrides]

`generation` times whole rollouts (`NewMDGenWrapper.inference`, the ODE solve
with the checkpoint's sampling method included) after a warm-up forward, and
counts model evaluations (NFE). `forward_profile` runs one
torch.profiler pass over a single model forward on the same inputs, after a
warm-up forward: kernel count and durations, GPU busy time within the forward
span, kernel-launch API calls on the CPU, and how many launches come from the
local attention and local IPA of the spatial groups. A forward whose GPU is idle
most of its span while it issues many short kernels is launch bound.
"""
from __future__ import annotations

import argparse
import bisect
from contextlib import contextmanager
import json
import math
import os
import time

import numpy as np
import torch

LAUNCH_PREFIXES = ('cudaLaunch', 'cuLaunch', 'cudaGraphLaunch')  # versioned names included
REGIONS = {'_spatial_attention': 'mdgen.local_attention', '_spatial_ipa': 'mdgen.local_ipa'}
MEMORY_OPS = ('Memcpy', 'Memset')
SHORT_KERNEL_US = 10.0


def _union_length(intervals):
    total, end = 0.0, -math.inf
    for start, stop in sorted(intervals):
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total


def kernel_stats(events, short_us=SHORT_KERNEL_US):
    """Summarize (name, device, start_us, end_us) records of one profiled forward.

    device is 'cuda' for GPU work and 'cpu' for host events; launch API calls and
    the `mdgen.*` region annotations are host events. Memcpy/memset GPU events are
    counted apart from kernels.
    """
    gpu = [(name, s, e) for name, device, s, e in events if device == 'cuda']
    kernels = [(s, e) for name, s, e in gpu if not name.startswith(MEMORY_OPS)]
    launches = sorted(s for name, device, s, _ in events
                      if device == 'cpu' and name.startswith(LAUNCH_PREFIXES))
    out = {'kernel_count': len(kernels), 'memory_op_count': len(gpu) - len(kernels),
           'launch_count': len(launches)}
    if kernels:
        durations = np.array([e - s for s, e in kernels])
        span = max(e for _, e in kernels) - min(s for s, _ in kernels)
        busy = _union_length(kernels)
        out.update({
            'kernel_time_us': float(durations.sum()),
            'kernel_span_us': float(span),
            'gpu_busy_us': float(busy),
            'gpu_busy_fraction': float(busy / span) if span else None,
            'kernel_median_us': float(np.median(durations)),
            f'kernel_fraction_under_{short_us:g}us': float((durations < short_us).mean()),
        })
    regions = {}
    for label in REGIONS.values():
        spans = [(s, e) for name, device, s, e in events if device == 'cpu' and name == label]
        # region ranges of one label do not overlap: count launches in each by bisection
        inside = sum(bisect.bisect_left(launches, e) - bisect.bisect_left(launches, s)
                     for s, e in spans)
        regions[label] = {
            'calls': len(spans),
            'host_time_us': float(sum(e - s for s, e in spans)),
            'launch_count': inside,
            'launch_fraction': inside / len(launches) if launches else None,
        }
    out['regions'] = regions
    return out


def profiler_records(profiler):
    records = []
    for event in profiler.events():
        if event.is_user_annotation and event.device_type == torch.autograd.DeviceType.CUDA:
            continue  # GPU-side copy of a record_function range, not GPU work
        device = 'cuda' if event.device_type == torch.autograd.DeviceType.CUDA else 'cpu'
        records.append((event.name, device, event.time_range.start, event.time_range.end))
    return records


@contextmanager
def annotate_spatial_regions(model):
    """Wrap each layer's spatial hooks in record_function labels for the profiler."""
    patched = []
    for module in model.modules():
        for method, label in REGIONS.items():
            if hasattr(type(module), method):
                original = getattr(module, method)

                def wrapped(*args, _original=original, _label=label, **kwargs):
                    with torch.profiler.record_function(_label):
                        return _original(*args, **kwargs)
                setattr(module, method, wrapped)
                patched.append((module, method))
    try:
        yield
    finally:
        for module, method in patched:
            delattr(module, method)


@contextmanager
def count_forwards(model):
    """Count model evaluations made through `forward_inference` (one per NFE)."""
    calls = [0]
    original = model.forward_inference

    def counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)
    model.forward_inference = counted
    try:
        yield calls
    finally:
        del model.forward_inference


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def partition_summary(partition, group_batch_size):
    cores = [g.core_indices.numel() for g in partition.groups]
    ghosts = [g.ghost_indices.numel() for g in partition.groups]
    return {'groups': len(cores), 'core_median': float(np.median(cores)),
            'ghost_median': float(np.median(ghosts)), 'group_batch_size': group_batch_size,
            'chunks_per_spatial_call': math.ceil(len(cores) / group_batch_size)}


@torch.no_grad()
def profile_generation(wrapper, batch, rollouts=1, profile_forward=True):
    """Time rollouts and count NFE; then profile one forward on the same inputs."""
    if any(torch.is_tensor(v) and v.shape[0] != 1 for v in batch.values()):
        raise ValueError('profile_generation runs one protein: every batch entry needs batch size 1')
    device = next(wrapper.parameters()).device
    cuda = device.type == 'cuda'
    _sync(device)
    start = time.perf_counter()
    prep = wrapper.prep_batch(batch)  # local mode: builds the frame-0 partitions
    _sync(device)
    kwargs = prep['model_kwargs']
    report = {'prep_s': time.perf_counter() - start,  # also paid inside every rollout
              'partition': partition_summary(kwargs['spatial_partitions'][0],
                                             wrapper.args.group_batch_size)}
    x = torch.randn_like(prep['latents'])
    t = torch.full((x.shape[0],), 0.5, device=device)
    wrapper.model.forward_inference(x, t, **kwargs)  # warm-up: allocator, cuDNN, JIT, lazy init
    _sync(device)

    runs = []
    for _ in range(rollouts):
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        start = time.perf_counter()
        with count_forwards(wrapper.model) as calls:
            wrapper.inference(batch)
        _sync(device)
        seconds = time.perf_counter() - start
        runs.append({'wall_clock_s': seconds, 'nfe': calls[0],
                     's_per_nfe': seconds / calls[0] if calls[0] else None,
                     'peak_memory_allocated_bytes':
                         torch.cuda.max_memory_allocated(device) if cuda else None})
    report['generation'] = {
        'sampling_method': wrapper.args.sampling_method, 'rollouts': runs,
        'wall_clock_s_median': float(np.median([r['wall_clock_s'] for r in runs])),
        'nfe_median': float(np.median([r['nfe'] for r in runs]))}

    start = time.perf_counter()
    wrapper.model.forward_inference(x, t, **kwargs)
    _sync(device)
    report['forward_s'] = time.perf_counter() - start
    if profile_forward:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with annotate_spatial_regions(wrapper.model), \
                torch.profiler.profile(activities=activities) as profiler:
            wrapper.model.forward_inference(x, t, **kwargs)
            _sync(device)
        profile = kernel_stats(profiler_records(profiler))
        calls = sum(region['calls'] for region in profile['regions'].values())
        report['partition']['spatial_calls_per_forward'] = calls
        report['partition']['group_chunks_per_forward'] = (
            calls * report['partition']['chunks_per_spatial_call'])
        if 'gpu_busy_us' in profile:  # against the unprofiled forward, free of profiler overhead
            profile['gpu_busy_fraction_of_forward'] = (
                profile['gpu_busy_us'] / (report['forward_s'] * 1e6))
        profile['top_device_ops'] = [
            {'name': e.key, 'count': e.count, 'self_device_time_us': e.self_device_time_total}
            for e in sorted(profiler.key_averages(), key=lambda e: -e.self_device_time_total)
            if e.device_type == torch.autograd.DeviceType.CUDA  # GPU rows only, no double count
            and not e.key.startswith('mdgen.')][:15] if cuda else []
        report['forward_profile'] = profile
    return report


def synthetic_batch(residues, seed):
    from ..train_step_probe import synthetic_items
    item = synthetic_items(argparse.Namespace(num_frames=1), residues, 1, seed)[0]
    return {'trans': item['trans'], 'rots': item['rots'], 'torsions': item['torsions'],
            'torsion_mask': item['torsion_mask'], 'seqres': item['seqres'], 'mask': item['mask'],
            'chain_id': torch.zeros(residues, dtype=torch.long),
            'residue_index': torch.arange(residues)}


def protein_batch(data_dir, name, suffix, seqres):
    """Frame 0 of `<data_dir>/<name><suffix>.npy`, as sim_inference conditions on it."""
    from ..geometry import atom14_to_atom37, atom14_to_frames, atom37_to_torsions
    from ..residue_constants import restype_order
    from ..topology import default_numbering, load_chain_numbering
    arr = np.asarray(np.load(f'{data_dir}/{name}{suffix}.npy', mmap_mode='r')[:1], dtype=np.float32)
    if arr.shape[1] != len(seqres):
        raise ValueError(f'{name}: NPY has {arr.shape[1]} residues, seqres {len(seqres)}')
    frames = atom14_to_frames(torch.from_numpy(arr))
    aatype = torch.tensor([restype_order[c] for c in seqres])
    atom37 = torch.from_numpy(atom14_to_atom37(arr, aatype[None])).float()
    torsions, torsion_mask = atom37_to_torsions(atom37, aatype[None])
    L = len(seqres)
    chain = (load_chain_numbering(f'{data_dir}/{name}{suffix}_chain.npz', L)
             or default_numbering(L))
    return {'trans': frames._trans, 'rots': frames._rots._rot_mats, 'torsions': torsions,
            'torsion_mask': torsion_mask[0], 'seqres': aatype, 'mask': torch.ones(L),
            **{key: torch.from_numpy(value) for key, value in chain.items()}}


def rollout_batch(first_frame, num_frames, device):
    """Batch of one, frame 0 repeated over num_frames (what rollouts condition on)."""
    out = {}
    for key, value in first_frame.items():
        value = value[None].to(device)
        if key in ('trans', 'rots', 'torsions'):
            value = value[:, :1].expand(-1, num_frames, *value.shape[2:])
        out[key] = value
    return out


def main(argv=None):
    from ..parsing import add_local_override_args, positive_int
    from ..train_step_probe import environment
    from .halo_error import load_wrapper
    # no abbreviations: the Slurm script detects --pdb_id/--synthetic_residues by name
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], allow_abbrev=False)
    parser.add_argument('--sim_ckpt', required=True)
    parser.add_argument('--output', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--synthetic_residues', type=positive_int)
    source.add_argument('--pdb_id')
    parser.add_argument('--data_dir')
    parser.add_argument('--split')
    parser.add_argument('--suffix', default='')
    parser.add_argument('--rollouts', type=positive_int, default=1)
    parser.add_argument('--num_frames', type=positive_int, default=None,
                        help='rollout length (default: the checkpoint num_frames)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ode_atol', type=float, default=1e-6)
    parser.add_argument('--ode_rtol', type=float, default=1e-3)
    parser.add_argument('--no_profile', action='store_true', help='skip torch.profiler')
    add_local_override_args(parser)
    args = parser.parse_args(argv)
    if args.pdb_id and not (args.data_dir and args.split):
        parser.error('--pdb_id needs --data_dir and --split')
    if not (0 < args.ode_atol < float('inf') and 0 < args.ode_rtol < float('inf')):
        parser.error('ODE tolerances must be positive and finite')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wrapper = load_wrapper(args.sim_ckpt, args, device)
    wrapper.args.ode_atol, wrapper.args.ode_rtol = args.ode_atol, args.ode_rtol
    num_frames = args.num_frames or wrapper.args.num_frames
    if getattr(wrapper.args, 'abs_time_emb', False) and num_frames != wrapper.args.num_frames:
        parser.error('fixed absolute time embeddings require the checkpoint num_frames')
    if args.synthetic_residues:
        first = synthetic_batch(args.synthetic_residues, args.seed)
    else:
        import pandas as pd
        df = pd.read_csv(args.split, index_col='name')
        if args.pdb_id not in df.index:
            parser.error(f'--pdb_id {args.pdb_id} not in --split')
        first = protein_batch(args.data_dir, args.pdb_id, args.suffix, df.seqres[args.pdb_id])
    torch.manual_seed(args.seed)
    batch = rollout_batch(first, num_frames, device)
    report = {
        'probe': 'generation-profile', 'num_frames': num_frames,
        'settings': {**vars(args), 'model': {key: getattr(wrapper.args, key, None) for key in (
            'num_frames', 'num_layers', 'prepend_ipa', 'sampling_method', 'spatial_cell_width',
            'spatial_contact_radius', 'core_capacity', 'ghost_capacity', 'group_batch_size',
            'cell_summaries')}},
        'residues': int(batch['seqres'].shape[1]),
        **profile_generation(wrapper, batch, rollouts=args.rollouts,
                             profile_forward=not args.no_profile),
        'environment': environment(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    return report


if __name__ == '__main__':
    main()
