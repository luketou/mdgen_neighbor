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
import json
import os
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
    """Median and max over finite values; `nonfinite` counts the rest (zero reference)."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {'count': int(len(finite)), 'nonfinite': int(len(values) - len(finite)),
            'median': float(np.median(finite)) if len(finite) else None,
            'max': float(finite.max()) if len(finite) else None}


@torch.no_grad()
def halo_errors(wrapper, batch, partition, ca, *, region_residues, halo_rings, regions=1,
                times=(0.5,), seed=0):
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
    for r in range(regions):
        try:
            selection = sample_region_groups(partition, ca, region_residues=region_residues,
                                             halo_rings=max(halo_rings),
                                             generator=torch.Generator().manual_seed(seed + r))
        except ValueError as error:
            skipped.append({'seed': seed + r, 'reason': str(error)})
            continue
        selections.setdefault(selection.loss_groups, selection)
    if not selections:
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
                'halo_groups': len(region.partition.groups) - len(selection.loss_groups),
                'region_residues': len(index), 'loss_residues': int(loss.sum()), 'times': []}
            cases.append((k, entry, index, loss, [ring_of_group[int(g)] for g in owner],
                          region.partition.to(device)))
    generator = torch.Generator().manual_seed(seed)
    residues = {k: [] for k in halo_rings}
    for t in times:
        x0 = torch.randn(x1.shape, generator=generator).to(device)
        tt = torch.full((1,), float(t), device=device)
        _, xt, _ = wrapper.transport.path_sampler.plan(tt, x0, x1)
        full = wrapper.model(xt, tt, **full_prep['model_kwargs'])
        for k, entry, index, loss, rings, sub in cases:
            prep = wrapper.prep_batch({**select_residues(batch, index, batched=True), 'partition': [sub]})
            out = wrapper.model(xt[:, :, index], tt, **prep['model_kwargs'])
            supervised = prep['loss_mask'][0][unconditioned][:, loss]  # unsupervised torsions drop out
            reference = full[0][unconditioned][:, index[loss]] * supervised
            diff = out[0][unconditioned][:, loss] * supervised - reference  # [T', n_loss, C]
            l2 = diff.double().pow(2).sum((0, 2)).sqrt().cpu()
            relative = l2 / reference.double().pow(2).sum((0, 2)).sqrt().cpu()
            residues[k].extend(zip(rings, l2.tolist(), relative.tolist()))
            entry['times'].append({'t': float(t), 'l2': stats(l2), 'relative': stats(relative)})
    return {'regions': region_reports, 'skipped_seeds': skipped, 'residues': residues}


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


def load_wrapper(path, cli_args, device):
    from ..parsing import apply_local_overrides
    from ..wrapper import NewMDGenWrapper
    from .checkpoint import load_compatible_state_dict
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model_args = copy.deepcopy(checkpoint['hyper_parameters']['args'])
    apply_local_overrides(model_args, cli_args)
    model_args.local_spatial = True
    unsupported = [flag for flag in ('abs_pos_emb', 'hyena', 'design', 'no_frames')
                   if getattr(model_args, flag, False)]
    if unsupported:  # same restrictions as --training_region
        raise ValueError(f'{path}: Training regions do not support ' + ', '.join(unsupported))
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
    add_local_override_args(parser)
    args = parser.parse_args(argv)
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wrapper = load_wrapper(args.sim_ckpt, args, device)
    dataset = protein_dataset(args, wrapper.args)
    proteins, residues = [], {k: [] for k in args.halo_rings}
    for idx, name in enumerate(dataset.df.index):
        if args.pdb_id and names_by_row[idx] not in args.pdb_id:
            continue
        item = dataset[idx]
        arr = np.lib.format.open_memmap(f'{args.data_dir}/{name}{args.suffix}.npy', 'r')
        partition, ca, _ = dataset.full_partition(name, arr)
        batch = {key: value.to(device) if torch.is_tensor(value) else value
                 for key, value in collate([item]).items()}
        result = halo_errors(wrapper, batch, partition, ca,
                             region_residues=args.region_residues, halo_rings=args.halo_rings,
                             regions=args.regions, times=args.times, seed=args.seed)
        for k in args.halo_rings:
            residues[k].extend(result['residues'][k])
        proteins.append({'name': name, 'residues': len(item['seqres']),
                         'groups': len(partition.groups), 'regions': result['regions'],
                         'skipped_seeds': result['skipped_seeds'],
                         'summary': summarize(result['residues'])})
        print(name, json.dumps(proteins[-1]['summary']), flush=True)
    if not proteins:
        parser.error('no protein in --split matches --pdb_id')
    report = {'experiment': 'halo-error', 'settings': {
                  **{k: v for k, v in vars(args).items()},
                  'model': {key: getattr(wrapper.args, key, None) for key in (
                      'num_frames', 'num_layers', 'spatial_cell_width', 'spatial_contact_radius',
                      'core_capacity', 'ghost_capacity', 'cell_summaries', 'frame_interval')}},
              'summary': summarize(residues), 'proteins': proteins,
              'environment': environment()}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    return report


if __name__ == '__main__':
    main()
