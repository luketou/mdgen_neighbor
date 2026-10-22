"""Offline full-protein partitions and the ghost drift report.

Usage:
    python -m mdgen.full_length.partition_store --split S --data_dir D --out_dir O \
        [--atlas --replicates 1 2 3] [--suffix _i40] [--drift_stride 10] \
        [--spatial_cell_width ... --spatial_contact_radius ... --core_capacity ... --ghost_capacity ...]

For every replicate NPY `<stem>.npy` this partitions the first frame with the
generation parameters, saves `<stem>_partition.npz`, and writes
`<stem>_drift.json`: on sampled frames, the fraction of residues within
contact_radius of a group's core that are not in that group's core or ghosts.
Frame 0 is non-zero only by ghost_capacity truncation (`frame0_spatial_truncated`),
up to cdist rounding for residues within ~1e-5 Å of contact_radius.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os

import numpy as np
import pandas as pd
import torch

from ..topology import default_numbering, load_chain_numbering
from .partition import (GroupDiagnostics, LocalGroup, PartitionParams, SpatialPartition,
                        build_spatial_partition)

DIAGNOSTIC_FIELDS = ('core_count', 'ghost_count', 'mandatory_neighbor_count',
                     'mandatory_retained_count', 'spatial_neighbor_count',
                     'spatial_retained_count')


def _concat(tensors):
    lengths = [len(t) for t in tensors]
    values = np.concatenate([t.cpu().numpy() for t in tensors]) if tensors else np.zeros(0)
    return values.astype(np.int64), np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)


def save_partition(path, partition, params, *, chain_id, residue_index, ca):
    """Store groups with the params, chain numbering and frame-0 Cα they were built from."""
    groups = partition.groups
    core, core_offsets = _concat([g.core_indices for g in groups])
    ghost, ghost_offsets = _concat([g.ghost_indices for g in groups])
    np.savez(
        path, core=core, core_offsets=core_offsets, ghost=ghost, ghost_offsets=ghost_offsets,
        cells=np.array([g.cell for g in groups], dtype=np.int64).reshape(-1, 3),
        diagnostics=np.array([[getattr(g.diagnostics, f) for f in DIAGNOSTIC_FIELDS]
                              for g in groups], dtype=np.int64).reshape(-1, len(DIAGNOSTIC_FIELDS)),
        owner_group=partition.owner_group.cpu().numpy(),
        residue_mask=partition.residue_mask.cpu().numpy(),
        chain_id=np.asarray(chain_id, dtype=np.int64),
        residue_index=np.asarray(residue_index, dtype=np.int64),
        ca=np.asarray(ca, dtype=np.float32),
        params=json.dumps(asdict(params), sort_keys=True))


def load_partition(path, params, *, chain_id, residue_index, ca):
    """Load a saved partition; raise if params, numbering or frame-0 Cα no longer match."""
    with np.load(path) as saved:
        stored = json.loads(str(saved['params']))
        mismatched = [k for k, v in asdict(params).items() if stored.get(k) != v]
        if mismatched:
            raise ValueError(f'{path}: built with different ' + ', '.join(
                f'{k} ({stored.get(k)} != {getattr(params, k)})' for k in mismatched))
        chain_id, residue_index = np.asarray(chain_id), np.asarray(residue_index)
        if len(chain_id) != len(saved['chain_id']):
            raise ValueError(f'{path}: built for {len(saved["chain_id"])} residues, '
                             f'got {len(chain_id)}; rebuild the partition')
        if not (np.array_equal(chain_id, saved['chain_id'])
                and np.array_equal(residue_index, saved['residue_index'])):
            raise ValueError(f'{path}: chain numbering changed since the partition was built; '
                             'rebuild the partition')
        if not np.allclose(np.asarray(ca, dtype=np.float32), saved['ca'], atol=1e-3):
            raise ValueError(f'{path}: frame-0 coordinates changed since the partition was '
                             'built; rebuild the partition')
        core, ghost = torch.from_numpy(saved['core']), torch.from_numpy(saved['ghost'])
        co, go = saved['core_offsets'], saved['ghost_offsets']
        groups = tuple(
            LocalGroup(cell=tuple(int(c) for c in cell),
                       core_indices=core[co[i]:co[i + 1]], ghost_indices=ghost[go[i]:go[i + 1]],
                       diagnostics=GroupDiagnostics(*(int(d) for d in diagnostics)))
            for i, (cell, diagnostics) in enumerate(zip(saved['cells'], saved['diagnostics'])))
        return SpatialPartition(
            groups=groups, owner_group=torch.from_numpy(saved['owner_group']),
            residue_mask=torch.from_numpy(saved['residue_mask']),
            core_capacity=params.core_capacity, ghost_capacity=params.ghost_capacity,
            cell_width=params.cell_width)


def ghost_drift(partition, ca_frames, contact_radius, frame_indices=None):
    """Per frame: residues within contact_radius of a core but outside core ∪ ghosts."""
    ca_frames = torch.as_tensor(ca_frames, dtype=torch.float32)
    valid = torch.nonzero(partition.residue_mask.bool().cpu()).flatten()
    groups = []  # per group: core, non-core valid residues, and whether each is a ghost
    for group in partition.groups:
        core = group.core_indices.cpu()
        others = valid[~torch.isin(valid, core)]
        if len(others):
            groups.append((core, others, torch.isin(others, group.ghost_indices.cpu())))
    frames = []
    for position, ca in enumerate(ca_frames):
        missing = neighbors = 0
        for core, others, is_ghost in groups:
            near = torch.cdist(ca[core], ca[others]).amin(dim=0) <= contact_radius
            neighbors += int(near.sum())
            missing += int((near & ~is_ghost).sum())
        frames.append({
            'frame': int(frame_indices[position]) if frame_indices is not None else position,
            'missing': missing, 'neighbors': neighbors,
            'fraction': missing / neighbors if neighbors else 0.0})
    return {
        'frames': frames,
        'max_fraction': max((f['fraction'] for f in frames), default=0.0),
        'frame0_spatial_truncated': sum(g.diagnostics.spatial_truncated_count
                                        for g in partition.groups),
    }


def main(argv=None):
    from ..parsing import LOCAL_DEFAULTS, LOCAL_OVERRIDE_TYPES, positive_int
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--split', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--suffix', default='')
    parser.add_argument('--atlas', action='store_true')
    parser.add_argument('--replicates', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--drift_stride', type=positive_int, default=10)
    for key in ('spatial_cell_width', 'spatial_contact_radius', 'core_capacity', 'ghost_capacity'):
        parser.add_argument(f'--{key}', type=LOCAL_OVERRIDE_TYPES[key], default=LOCAL_DEFAULTS[key])
    args = parser.parse_args(argv)
    params = PartitionParams.from_args(args)
    os.makedirs(args.out_dir, exist_ok=True)

    summary = []
    df = pd.read_csv(args.split, index_col='name')
    for name in df.index:
        stems = ([f'{name}_R{i}{args.suffix}' for i in args.replicates] if args.atlas
                 else [f'{name}{args.suffix}'])
        for stem in stems:
            arr = np.load(f'{args.data_dir}/{stem}.npy', mmap_mode='r')
            L = arr.shape[1]
            if L != len(df.seqres[name]):
                raise ValueError(f'{stem}: NPY has {L} residues but seqres has '
                                 f'{len(df.seqres[name])}')
            frames = list(range(0, arr.shape[0], args.drift_stride))  # always includes frame 0
            # atom14 slot 1 is CA (Å); read only the sampled frames
            ca = torch.from_numpy(np.asarray(arr[::args.drift_stride, :, 1], dtype=np.float32))
            numbering = (load_chain_numbering(f'{args.data_dir}/{stem}_chain.npz', L)
                         or default_numbering(L))
            partition = build_spatial_partition(
                ca[0], mask=torch.ones(L, dtype=torch.bool),
                chain_id=torch.from_numpy(numbering['chain_id']),
                residue_index=torch.from_numpy(numbering['residue_index']),
                **params.partition_kwargs())
            save_partition(f'{args.out_dir}/{stem}_partition.npz', partition, params,
                           chain_id=numbering['chain_id'],
                           residue_index=numbering['residue_index'], ca=ca[0])
            drift = ghost_drift(partition, ca, params.contact_radius, frames)
            with open(f'{args.out_dir}/{stem}_drift.json', 'w') as f:
                json.dump({'stem': stem, 'params': asdict(params),
                           'groups': len(partition.groups), **drift}, f, indent=2)
            summary.append({'stem': stem, 'groups': len(partition.groups),
                            'max_fraction': drift['max_fraction'],
                            'frame0_spatial_truncated': drift['frame0_spatial_truncated']})
            print(stem, summary[-1], flush=True)
    with open(f'{args.out_dir}/summary.json', 'w') as f:
        json.dump({'params': asdict(params), 'replicates': summary}, f, indent=2)


if __name__ == '__main__':
    main()
