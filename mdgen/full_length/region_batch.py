"""Training region field selection and padded item assembly.

Residue axes belong here so trajectory loading, synthetic probes and halo
comparisons use the same field layout. Geometry conversion stays with the
trajectory reader; halo comparisons select collated fields without padding.
"""
from dataclasses import replace

import numpy as np
import torch

from .group_region import region_subpartition, sample_region_groups


def sample_region(partition, ca, args, generator=None):
    """(selection, region partition) for one Training region of a full-protein partition."""
    selection = sample_region_groups(partition, ca, region_residues=args.region_residues,
                                     halo_rings=args.halo_rings, generator=generator,
                                     seed_group=getattr(args, 'region_seed_group', None))
    return selection, region_subpartition(partition, selection)


_RESIDUE_AXES = {'torsions': 1, 'trans': 1, 'rots': 1, 'torsion_mask': 0, 'seqres': 0,
                 'mask': 0, 'chain_id': 0, 'residue_index': 0}


def select_residues(item, residues, *, batched=False):
    """Select known residue fields; leave names and other metadata unchanged.

    Collated fields have one extra leading batch dimension. NumPy fields stay
    NumPy; tensor fields keep their dtype and device.
    """
    residues = torch.as_tensor(residues, dtype=torch.long)
    out = {}
    for key, value in item.items():
        if key not in _RESIDUE_AXES:
            out[key] = value
            continue
        axis = _RESIDUE_AXES[key] + int(batched)
        if isinstance(value, np.ndarray):
            out[key] = np.take(value, residues.cpu().numpy(), axis=axis)
        else:
            out[key] = value.index_select(axis, residues.to(value.device))
    return out


def assemble_region(item, selection, region, width, *, source_residues=None):
    """Select and pad a Training region from converted per-residue fields.

    By default fields use full-protein indexing. A trajectory reader can pass
    source_residues when it converted only the region and its predecessors;
    selection then uses positions within that smaller source tensor.
    """
    residues = region.residues
    if source_residues is not None:
        residues = np.searchsorted(source_residues, residues.cpu().numpy())
    return _pad_region(select_residues(item, residues), selection, region, width)


def _pad_region(item, selection, region, width):
    """Pad a region's per-residue fields (in region order) to width residues.

    Adds the padded sub-partition (whose `position_index` holds full-protein
    indices), `loss_core_mask` (residues owned by a loss group) and a `region`
    summary. Padding is masked, ownerless and outside the loss. Group indices
    share one storage, so a DataLoader worker sends one tensor, not two per group.
    """
    n = len(region.residues)
    pad = width - n

    def padded(value, axis):
        if isinstance(value, np.ndarray):
            return np.concatenate([value, np.zeros((pad, *value.shape[1:]), dtype=value.dtype)])
        shape = list(value.shape)
        shape[axis] = pad
        return torch.cat([value, value.new_zeros(shape)], axis)

    out = {key: padded(value, _RESIDUE_AXES[key]) if key in _RESIDUE_AXES else value
           for key, value in item.items()}
    out['rots'][:, n:] = torch.eye(3)
    partition = region.partition
    groups = [index for g in partition.groups for index in (g.core_indices, g.ghost_indices)]
    views = torch.cat(groups).split([len(index) for index in groups])
    out['partition'] = replace(
        partition,
        groups=tuple(replace(g, core_indices=views[2 * i], ghost_indices=views[2 * i + 1])
                     for i, g in enumerate(partition.groups)),
        owner_group=torch.cat([partition.owner_group, partition.owner_group.new_full((pad,), -1)]),
        residue_mask=padded(partition.residue_mask, 0),
        position_index=padded(partition.position_index, 0))
    out['loss_core_mask'] = padded(region.loss_core_mask.numpy().astype(np.float32), 0)
    out['region'] = {
        'loss_groups': len(selection.loss_groups),
        'loss_group_ids': list(selection.loss_groups),
        'halo_group_ids': list(selection.halo_groups),
        'halo_groups': len(selection.halo_groups),
        'residues': n,
        'loss_residues': int(region.loss_core_mask.sum()),
        'next_region_size': selection.next_region_size,
    }
    return out
