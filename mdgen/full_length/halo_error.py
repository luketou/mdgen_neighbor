"""Halo approximation error of Training regions against the full protein (#12 Ticket 5).

Usage:
    python -m mdgen.full_length.halo_error --sim_ckpt atlas.ckpt --data_dir D --split S \
        --output report.json [--pdb_id NAME ...] [--replicate 1] [--suffix _i40] \
        [--partition_dir P] [--region_residues 1024] [--halo_rings 1 2] [--regions 4] \
        [--times 0.5] [--seed 0] [--spatial_cell_width ... other local overrides]

For each protein one full-protein forward and every region forward share the
noise, t and loss groups. Loss groups are chosen by `sample_region_groups` with
the deepest --halo_rings, so every setting fits --region_residues; each setting
then adds its own halo around the same loss groups. Only loss-core outputs on
unconditioned frames are compared. Per residue: L2 of the output difference over
frames and channels, and that L2 relative to the full-protein output's L2.
Residues are grouped by the ring of their loss group from the region boundary
(0 = a loss group with a ghost owned outside the loss groups).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from .group_region import region_subpartition, sample_region_groups, with_halo
from .region_batch import select_residues


def boundary_rings(partition, loss_groups):
    """Ring of each loss group from the region boundary; None if nothing lies outside.

    Ring 0 groups have a ghost owned by a group outside loss_groups; ring r groups
    have a ghost owned by a ring r - 1 group. None: no ghost-owner path leads outside
    (e.g. every group is a loss group).
    """
    owner = partition.owner_group.cpu()
    loss = set(loss_groups)
    sources = {g: set(owner[partition.groups[g].ghost_indices.cpu()].tolist()) - {-1, g}
               for g in loss}
    ring = {g: 0 for g in loss if sources[g] - loss}
    frontier = set(ring)
    while frontier:
        frontier = {g for g in loss - set(ring) if sources[g] & frontier}
        depth = max(ring.values()) + 1
        ring.update({g: depth for g in frontier})
    return {g: ring.get(g) for g in sorted(loss)}


def stats(values):
    """Finite distribution; undefined ratios are separate from NaN/Inf model values."""
    values = list(values)
    undefined = sum(v is None for v in values)
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {'count': int(len(finite)), 'nonfinite': int(len(values) - len(finite)),
            'undefined': undefined,
            'median': float(np.median(finite)) if len(finite) else None,
            'p95': float(np.percentile(finite, 95, method='linear')) if len(finite) else None,
            'max': float(finite.max()) if len(finite) else None}


def measured_forward(call, device, warmup, repeats):
    """Synchronized observations of the same input; live reference memory is included."""
    cuda = torch.device(device).type == 'cuda'
    for _ in range(warmup):
        call()
    measurements = []
    output = None
    for _ in range(repeats):
        output = None
        if cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        output = call()
        if cuda:
            torch.cuda.synchronize(device)
        measurements.append({
            'time_s': time.perf_counter() - start,
            'peak_memory_allocated_bytes': torch.cuda.max_memory_allocated(device) if cuda else None,
            'peak_memory_reserved_bytes': torch.cuda.max_memory_reserved(device) if cuda else None})
    return output, measurements


def seed_region_report(partition, halo_rings, budget):
    """Every seed group is counted, including those outside the budget."""
    report = {}
    for k in halo_rings:
        sizes = [len(with_halo(partition, (g,), k).residues)
                 for g in range(len(partition.groups))]
        report[str(k)] = {
            'sizes': sizes, 'min': min(sizes), **stats(sizes),
            'feasible': sum(n <= budget for n in sizes), 'total': len(sizes),
            'infeasible': [{'seed_group': g, 'residues': n, 'reason': 'seed_exceeds_budget'}
                           for g, n in enumerate(sizes) if n > budget]}
    return report


def observation_stats(rows):
    """Preserve undefined denominators separately from nonfinite model values."""
    classes = ('regular', 'zero_reference', 'near_zero_reference', 'no_supervision', 'nonfinite')
    return {
        'samples': len(rows),
        **{key: sum(row['reference_class'] == key for row in rows) for key in classes},
        **{key: stats([row[key] for row in rows if row[key] is not None])
           for key in ('l2', 'reference_norm', 'absolute_rms', 'relative')}}


def stratify(observations):
    keys = ('t', 'halo_rings', 'boundary_ring', 'truncated')
    groups = {}
    for row in observations:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    return [{**dict(zip(keys, key)), **observation_stats(rows)} for key, rows in groups.items()]


@torch.no_grad()
def halo_errors(wrapper, batch, partition, ca, *, region_residues, halo_rings, regions=1,
                times=(0.5,), seed=0, warmup=0, repeats=1, seed_groups=None):
    """Per-residue loss-core errors of each halo setting; batch is one collated protein.

    Returns {'regions': [...], 'skipped_seeds': [...], 'residues': {k: [(ring, l2, relative), ...]}}.
    Seeds whose seed group with the deepest halo exceeds region_residues are skipped;
    raises ValueError when no seed fits.
    """
    if any(torch.is_tensor(v) and v.shape[0] != 1 for v in batch.values()) or len(batch['name']) != 1:
        raise ValueError('halo_errors compares one protein: every batch entry needs batch size 1')
    device = batch['trans'].device
    full_prep = wrapper.prep_batch({**batch, 'partition': [partition.to(device)]})
    x1 = full_prep['latents']
    unconditioned = full_prep['model_kwargs']['x_cond_mask'][0, :, 0] == 0  # [T]
    selections = {}  # distinct loss-group sets; two seeds can grow the same region
    skipped = []  # seeds whose seed group with the deepest halo alone exceeds the budget
    candidates = []
    for r in range(regions if seed_groups is None else len(seed_groups)):
        candidate = {'seed': seed + r} if seed_groups is None else {'seed_group': seed_groups[r]}
        try:
            selection = sample_region_groups(partition, ca, region_residues=region_residues,
                                             halo_rings=max(halo_rings),
                                             generator=torch.Generator().manual_seed(seed + r),
                                             **({'seed_group': seed_groups[r]}
                                                if seed_groups is not None else {}))
        except ValueError as error:
            skipped.append({**candidate, 'reason': str(error)})
            candidates.append({**candidate, 'status': 'infeasible', 'reason': str(error)})
            continue
        duplicate = selection.loss_groups in selections
        candidates.append({**candidate, 'status': 'duplicate' if duplicate else 'selected',
                           'loss_groups': list(selection.loss_groups)})
        selections.setdefault(selection.loss_groups, selection)
    if not selections and seed_groups is None:
        raise ValueError('no region fits: ' + '; '.join(s['reason'] for s in skipped))
    cases = []  # (k, report entry, region residues, loss mask, ring per loss residue, partition)
    region_reports = []
    for selection in selections.values():
        ring_of_group = boundary_rings(partition, selection.loss_groups)
        report = {'loss_groups': list(selection.loss_groups), 'by_halo': {}}
        region_reports.append(report)
        for k in halo_rings:
            region = region_subpartition(partition, with_halo(partition, selection.loss_groups, k))
            index, loss = region.residues.to(device), region.loss_core_mask.to(device)
            owner = partition.owner_group.cpu()[region.residues[region.loss_core_mask]]
            entry = report['by_halo'][str(k)] = {
                'region_id': len(region_reports) - 1,
                'loss_groups': list(selection.loss_groups),
                'halo_groups': len(region.partition.groups) - len(selection.loss_groups),
                'region_residues': len(index), 'loss_residues': int(loss.sum()),
                'truncated': len(index) < partition.residue_mask.numel(), 'times': []}
            cases.append((k, entry, index, loss, [ring_of_group[int(g)] for g in owner],
                          region.partition.to(device)))
    generator = torch.Generator().manual_seed(seed)
    residues = {k: [] for k in halo_rings}
    measurements = {'full': [], 'regions': []}
    observations = []
    nonfinite_counts = {'input': sum(int((~v.isfinite()).sum()) for v in batch.values()
                                     if torch.is_tensor(v)), 'full_output': 0, 'region_output': 0}
    for t in times if selections else ():
        x0 = torch.randn(x1.shape, generator=generator).to(device)
        tt = torch.full((1,), float(t), device=device)
        _, xt, _ = wrapper.transport.path_sampler.plan(tt, x0, x1)
        full, measured = measured_forward(
            lambda: wrapper.model(xt, tt, **full_prep['model_kwargs']), device, warmup, repeats)
        measurements['full'].append({'t': float(t), 'runs': measured})
        nonfinite_counts['full_output'] += int((~full.isfinite()).sum())
        for k, entry, index, loss, rings, sub in cases:
            prep = wrapper.prep_batch({**select_residues(batch, index, batched=True), 'partition': [sub]})
            out, measured = measured_forward(
                lambda: wrapper.model(xt[:, :, index], tt, **prep['model_kwargs']),
                device, warmup, repeats)
            measurements['regions'].append({'t': float(t), 'halo_rings': k,
                                            'loss_groups': entry.get('loss_groups'),
                                            'region_residues': len(index), 'runs': measured})
            nonfinite_counts['region_output'] += int((~out.isfinite()).sum())
            supervised = prep['loss_mask'][0][unconditioned][:, loss]  # unsupervised torsions drop out
            reference = full[0][unconditioned][:, index[loss]] * supervised
            diff = out[0][unconditioned][:, loss] * supervised - reference  # [T', n_loss, C]
            l2 = diff.double().pow(2).sum((0, 2)).sqrt().cpu()
            refnorm = reference.double().pow(2).sum((0, 2)).sqrt().cpu()
            components = supervised.expand_as(reference).double().sum((0, 2)).cpu()
            for i, (a, ref, n, ring) in enumerate(zip(l2.tolist(), refnorm.tolist(),
                                                   components.tolist(), rings)):
                finite = np.isfinite(a) and np.isfinite(ref)
                category = ('nonfinite' if not finite else 'no_supervision' if n == 0 else
                            'zero_reference' if ref == 0 else
                            'near_zero_reference' if ref / np.sqrt(n) <= 1e-6 else 'regular')
                observations.append({
                    'region_id': entry['region_id'],
                    'residue_index': int(index[loss][i]), 't': float(t), 'halo_rings': k,
                    'boundary_ring': ring, 'truncated': entry['truncated'],
                    'effective_components': int(n), 'reference_class': category,
                    'l2': a if finite and n else None,
                    'reference_norm': ref if finite and n else None,
                    'absolute_rms': a / np.sqrt(n) if finite and n else None,
                    'relative': a / ref if category == 'regular' else None})
            relative_values = [row['relative'] if row['reference_class'] != 'nonfinite'
                               else float('nan') for row in observations[-len(rings):]]
            residues[k].extend(zip(rings, l2.tolist(), relative_values))
            entry['times'].append({'t': float(t), 'l2': stats(l2), 'relative': stats(relative_values)})
    loss_union = {int(i) for s in selections.values() for g in s.loss_groups
                  for i in partition.groups[g].core_indices}
    return {'regions': region_reports, 'skipped_seeds': skipped, 'residues': residues,
            'measurements': measurements, 'observations': observations,
            'nonfinite_counts': nonfinite_counts,
            'strata': stratify(observations), 'candidates': candidates,
            'coverage': {'attempted': len(candidates), 'feasible': len(candidates) - len(skipped),
                         'skipped': len(skipped), 'unique_regions': len(selections),
                         'duplicates': sum(c['status'] == 'duplicate' for c in candidates),
                         'loss_residue_fraction': len(loss_union) / partition.residue_mask.numel()}}


def summarize(residues):
    """Relative and L2 error per halo setting, overall and by boundary ring."""
    out = {}
    for k, rows in residues.items():
        rings = sorted({ring for ring, _, _ in rows}, key=lambda r: (r is None, r or 0))
        out[str(k)] = {
            'l2': stats([l2 for _, l2, _ in rows]),
            'relative': stats([rel for _, _, rel in rows]),
            'by_ring': {str(ring): {
                'l2': stats([l2 for r, l2, _ in rows if r == ring]),
                'relative': stats([rel for r, _, rel in rows if r == ring])}
                for ring in rings},
        }
    return out


def checkpoint_args(checkpoint, cli_args):
    from ..parsing import apply_local_overrides
    model_args = copy.deepcopy(checkpoint['hyper_parameters']['args'])
    apply_local_overrides(model_args, cli_args)
    model_args.local_spatial = True
    if getattr(cli_args, 'num_frames', None) is not None:
        model_args.num_frames = cli_args.num_frames
    model_args.grad_checkpointing = False
    model_args.dropout = 0.0
    unsupported = [flag for flag in ('abs_pos_emb', 'hyena', 'design', 'no_frames')
                   if getattr(model_args, flag, False)]
    if unsupported:  # same restrictions as --training_region
        raise ValueError('Training regions do not support ' + ', '.join(unsupported))
    return model_args


def load_wrapper(path, cli_args, device):
    from ..wrapper import NewMDGenWrapper
    from .checkpoint import load_compatible_state_dict
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model_args = checkpoint_args(checkpoint, cli_args)
    wrapper = NewMDGenWrapper(model_args)
    report = load_compatible_state_dict(wrapper, checkpoint)
    missing = [name for name in report['newly_initialized'] if '.cell_summaries.' not in name]
    if missing or report['shape_mismatch'] or report['unexpected']:
        raise ValueError(f'Checkpoint incompatible: {report}')
    return wrapper.eval().float().to(device)


def protein_dataset(cli_args, model_args):
    """Full proteins, frame window starting at frame 0 (where partitions are built)."""
    from ..dataset import MDGenDataset
    args = SimpleNamespace(
        data_dir=cli_args.data_dir, suffix=cli_args.suffix, partition_dir=cli_args.partition_dir,
        num_frames=model_args.num_frames, overfit=False,
        frame_interval=getattr(model_args, 'frame_interval', None),  # the trained time stride
        overfit_peptide=None, overfit_frame=True, copy_frames=False, no_frames=False,
        atlas=False, crop=0, training_region=False,
        **{key: getattr(model_args, key) for key in (
            'spatial_cell_width', 'spatial_contact_radius', 'core_capacity', 'ghost_capacity')})
    dataset = MDGenDataset(args, cli_args.split)
    if cli_args.replicate is not None:  # read <name>_R<i> while keeping each row's seqres
        dataset.df.index = [f'{name}_R{cli_args.replicate}' for name in dataset.df.index]
    return dataset


def main(argv=None):
    from ..parsing import add_local_override_args, positive_int
    from ..train_step_probe import environment
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--sim_ckpt', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pdb_id', nargs='*', default=[])
    parser.add_argument('--replicate', type=int, default=None)
    parser.add_argument('--suffix', default='')
    parser.add_argument('--partition_dir', default=None)
    parser.add_argument('--region_residues', type=positive_int, default=1024)
    parser.add_argument('--halo_rings', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--regions', type=positive_int, default=4)
    parser.add_argument('--times', type=float, nargs='+', default=[0.5])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_frames', type=positive_int, default=None)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--repeats', type=positive_int, default=1)
    parser.add_argument('--inventory_only', action='store_true',
                        help='CPU partition statistics only; does not instantiate a model')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--seed_groups', type=int, nargs='+')
    selection.add_argument('--selection_seed', type=int)
    add_local_override_args(parser)
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error('--warmup must be non-negative')
    if min(args.halo_rings) < 0:
        parser.error('--halo_rings must be non-negative')
    if not all(0 < t < 1 for t in args.times):
        parser.error('--times must lie in (0, 1)')
    args.halo_rings = sorted(set(args.halo_rings))
    import pandas as pd
    names_by_row = list(pd.read_csv(args.split, index_col='name').index)
    if unknown := sorted(set(args.pdb_id) - set(names_by_row)):
        parser.error(f'--pdb_id not in --split: {", ".join(unknown)}')

    from ..dataset import collate
    device = 'cuda' if torch.cuda.is_available() and not args.inventory_only else 'cpu'
    from ..precision import configure_precision
    if args.inventory_only:
        model_args = checkpoint_args(torch.load(args.sim_ckpt, map_location='cpu',
                                               weights_only=False), args)
    else:
        configure_precision('fp32-reference')
        wrapper = load_wrapper(args.sim_ckpt, args, device)
        model_args = wrapper.args
    dataset = protein_dataset(args, model_args)
    proteins, residues = [], {k: [] for k in args.halo_rings}
    for idx, name in enumerate(dataset.df.index):
        if args.pdb_id and names_by_row[idx] not in args.pdb_id:
            continue
        arr = np.lib.format.open_memmap(f'{args.data_dir}/{name}{args.suffix}.npy', 'r')
        partition, ca, _ = dataset.full_partition(name, arr)
        if args.inventory_only:
            proteins.append({'name': name, 'residues': len(ca), 'groups': len(partition.groups),
                             'seed_regions': seed_region_report(partition, args.halo_rings,
                                                                args.region_residues),
                             'candidate_seed_groups': torch.randperm(len(partition.groups),
                                 generator=torch.Generator().manual_seed(args.selection_seed or 43002)
                                 )[:args.regions].tolist()})
            continue
        item = dataset[idx]
        batch = {key: value.to(device) if torch.is_tensor(value) else value
                 for key, value in collate([item]).items()}
        result = halo_errors(wrapper, batch, partition, ca,
                             region_residues=args.region_residues, halo_rings=args.halo_rings,
                             regions=args.regions, times=args.times, seed=args.seed,
                             warmup=args.warmup, repeats=args.repeats,
                             seed_groups=(torch.randperm(len(partition.groups), generator=
                                 torch.Generator().manual_seed(args.selection_seed))[:args.regions].tolist()
                                 if args.selection_seed is not None else args.seed_groups))
        for k in args.halo_rings:
            residues[k].extend(result['residues'][k])
        proteins.append({'name': name, 'residues': len(item['seqres']),
                         'groups': len(partition.groups), 'regions': result['regions'],
                         'skipped_seeds': result['skipped_seeds'],
                         'batch_shapes': {'trans': list(batch['trans'].shape)},
                         'measurements': result['measurements'],
                         'observations': result['observations'], 'strata': result['strata'],
                         'nonfinite_counts': result['nonfinite_counts'],
                         'candidates': result['candidates'], 'coverage': result['coverage'],
                         'seed_regions': seed_region_report(partition, args.halo_rings,
                                                            args.region_residues),
                         'summary': summarize(result['residues'])})
        print(name, json.dumps(proteins[-1]['summary']), flush=True)
    if not proteins:
        parser.error('no protein in --split matches --pdb_id')
    report = {'experiment': 'halo-error', 'settings': {
                  **{k: v for k, v in vars(args).items()},
                  'model': {key: getattr(model_args, key, None) for key in (
                      'num_frames', 'num_layers', 'spatial_cell_width', 'spatial_contact_radius',
                      'core_capacity', 'ghost_capacity', 'cell_summaries', 'frame_interval',
                      'grad_checkpointing', 'ffn_chunk_size', 'group_batch_size')}},
              'summary': summarize(residues), 'proteins': proteins,
              'environment': environment()}
    if not args.inventory_only:
        report['precision_state'] = {
                  'precision_config': 'fp32-reference',
                  'parameter_dtypes': sorted({str(p.dtype) for p in wrapper.parameters()}),
                  'float32_matmul_precision': torch.get_float32_matmul_precision(),
                  'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                  'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                  'autocast_enabled': torch.is_autocast_enabled(device)}
    with open(args.sim_ckpt, 'rb') as stream:
        report['checkpoint_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    return report


if __name__ == '__main__':
    main()
