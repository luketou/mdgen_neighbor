from dataclasses import replace
import torch
from torch.utils.data import default_collate
from .full_length.group_region import (largest_seed_region, region_subpartition,
                                       sample_region_groups)
from .full_length.partition import PartitionParams, build_spatial_partition
from .full_length.partition_store import load_partition
from .topology import default_numbering, load_chain_numbering
from .rigid_utils import Rigid
from .residue_constants import restype_order, restype_atom37_mask
import numpy as np
import pandas as pd
from .geometry import atom37_to_torsions, atom14_to_atom37, atom14_to_frames
       
class MDGenDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, repeat=1, region_seed=None, replicates=None):
        super().__init__()
        self.df = pd.read_csv(split, index_col='name')
        self.args = args
        self.repeat = repeat
        self.region_seed = region_seed  # fixed Training regions, e.g. for validation
        # --atlas replicate numbers; older saved args lack --replicates
        self.replicates = replicates or getattr(args, 'replicates', [1, 2, 3])
        self.partitions = {}  # frame-0 full-protein partition per replicate, built once
        if args.training_region and {'chain_id', 'residue_index'} & set(self.df.columns):
            raise ValueError(f'{split}: chain numbering now comes from <npy stem>_chain.npz '
                             '(scripts/prep_sims.py), not split CSV columns')
    def __len__(self):
        if self.args.overfit_peptide:
            return 1000
        return self.repeat * len(self.df)

    def __getitem__(self, idx):
        seed_index = idx  # before the modulo, so --val_repeat copies get distinct fixed regions
        idx = idx % len(self.df)
        if self.args.overfit:
            idx = 0

        if self.args.overfit_peptide is None:
            name = self.df.index[idx]
            seqres = self.df.seqres[name]
        else:
            name = self.args.overfit_peptide
            seqres = name

        rng = np.random
        if self.args.training_region and self.region_seed is not None:
            rng = np.random.RandomState(self.region_seed + seed_index)  # fix replica and window too

        if self.args.atlas:
            i = self.replicates[rng.randint(0, len(self.replicates))]
            full_name = f"{name}_R{i}"
        else:
            full_name = name
        arr = np.lib.format.open_memmap(f'{self.args.data_dir}/{full_name}{self.args.suffix}.npy', 'r')
        if self.args.frame_interval:
            arr = arr[::self.args.frame_interval]
        
        # data-preprocess-fixes:start
        available_frames = arr.shape[0]
        requested_frames = self.args.num_frames
        if available_frames < requested_frames:
            raise ValueError(
                f'{full_name}: requested {requested_frames} frames, '
                f'but NPY only contains {available_frames}'
            )

        max_start = available_frames - requested_frames
        if self.args.overfit_frame or max_start == 0:
            frame_start = 0
        else:
            frame_start = rng.randint(0, max_start + 1)
        end = frame_start + self.args.num_frames
        # data-preprocess-fixes:end
        region = None
        if self.args.training_region:  # pick residues before any frame or torsion conversion
            generator = None
            if self.region_seed is not None:
                generator = torch.Generator().manual_seed(self.region_seed + seed_index)
            partition, ca, chain = self.full_partition(full_name, arr)
            selection, region = sample_region(partition, ca, self.args, generator)
            residues = region.residues.numpy()
            # with each residue's predecessor, phi and pre-omega match the full protein
            ext = np.unique(np.concatenate([residues, np.maximum(residues - 1, 0)]))
            pick = np.searchsorted(ext, residues)
            arr = arr[frame_start:end, ext]
        else:
            arr = arr[frame_start:end]
        # arr = np.copy(arr[frame_start:end]) * 10 # convert to angstroms
        arr = np.copy(arr).astype(np.float32) # / 10.0 # convert to nm
        if self.args.copy_frames:
            arr[1:] = arr[0]

        # arr should be in ANGSTROMS
        frames = atom14_to_frames(torch.from_numpy(arr))
        seqres = np.array([restype_order[c] for c in seqres])
        if region is not None:
            seqres = seqres[ext]
        aatype = torch.from_numpy(seqres)[None].expand(self.args.num_frames, -1)
        atom37 = torch.from_numpy(atom14_to_atom37(arr, aatype)).float()
        
        L = frames.shape[1]
        mask = np.ones(L, dtype=np.float32)
        
        if self.args.no_frames:
            return {
                'name': full_name,
                'frame_start': frame_start,
                'atom37': atom37,
                'seqres': seqres,
                'mask': restype_atom37_mask[seqres], # (L, 37)
            }
        torsions, torsion_mask = atom37_to_torsions(atom37, aatype)
        
        torsion_mask = torsion_mask[0]

        if region is not None:
            return {
                'name': full_name,
                'frame_start': frame_start,
                **pad_region({
                    'torsions': torsions[:, pick],
                    'torsion_mask': torsion_mask[pick],
                    'trans': frames._trans[:, pick],
                    'rots': frames._rots._rot_mats[:, pick],
                    'seqres': seqres[pick],
                    'mask': mask[pick],
                    **{key: value[residues] for key, value in chain.items()},
                }, selection, region, self.args.region_residues),
            }

        chain = self.chain_numbering(full_name, L)
        if self.args.atlas:
            if L > self.args.crop:
                start = np.random.randint(0, L - self.args.crop + 1)
                torsions = torsions[:,start:start+self.args.crop]
                frames = frames[:,start:start+self.args.crop]
                seqres = seqres[start:start+self.args.crop]
                mask = mask[start:start+self.args.crop]
                torsion_mask = torsion_mask[start:start+self.args.crop]
                chain = {key: value[start:start+self.args.crop] for key, value in chain.items()}
                
            
            elif L < self.args.crop:
                pad = self.args.crop - L
                frames = Rigid.cat([
                    frames, 
                    Rigid.identity((self.args.num_frames, pad), requires_grad=False, fmt='rot_mat')
                ], 1)
                mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
                seqres = np.concatenate([seqres, np.zeros(pad, dtype=int)])
                torsions = torch.cat([torsions, torch.zeros((torsions.shape[0], pad, 7, 2), dtype=torch.float32)], 1)
                torsion_mask = torch.cat([torsion_mask, torch.zeros((pad, 7), dtype=torch.float32)])
                chain = {key: np.concatenate([value, np.zeros(pad, dtype=np.int64)])
                         for key, value in chain.items()}

        return {
            'name': full_name,
            'frame_start': frame_start,
            'torsions': torsions,
            'torsion_mask': torsion_mask,
            'trans': frames._trans,
            'rots': frames._rots._rot_mats,
            'seqres': seqres,
            'mask': mask, # (L,)
            **chain,  # every item carries numbering, so batches always collate
        }

    def chain_numbering(self, full_name, length):
        """<npy stem>_chain.npz written next to the NPY by scripts/prep_sims.py, else one chain."""
        path = f'{self.args.data_dir}/{full_name}{self.args.suffix}_chain.npz'
        return load_chain_numbering(path, length) or default_numbering(length)

    def full_partition(self, full_name, arr):
        """(partition, frame-0 Cα, chain numbering) of a replicate, cached: the partition
        stored by `partition_store` (checked to match), else built here."""
        if full_name not in self.partitions:
            chain = self.chain_numbering(full_name, arr.shape[1])
            params = PartitionParams.from_args(self.args)
            ca = frame0_ca(arr)
            if self.args.partition_dir:
                path = f'{self.args.partition_dir}/{full_name}{self.args.suffix}_partition.npz'
                partition = load_partition(path, params, ca=ca, **chain)
            else:
                partition = build_spatial_partition(
                    torch.from_numpy(ca), mask=torch.ones(len(ca), dtype=torch.bool),
                    **{key: torch.from_numpy(value) for key, value in chain.items()},
                    **params.partition_kwargs())
            self.partitions[full_name] = partition, torch.from_numpy(ca), chain
        return self.partitions[full_name]

    def check_region_budget(self):
        """Raise at startup, not mid-epoch, if some seed group cannot fit --region_residues."""
        if self.args.overfit_peptide is not None:
            return
        for name in self.df.index:
            for full_name in ([f'{name}_R{i}' for i in self.replicates] if self.args.atlas
                              else [name]):
                arr = np.lib.format.open_memmap(
                    f'{self.args.data_dir}/{full_name}{self.args.suffix}.npy', 'r')
                partition, _, _ = self.full_partition(full_name, arr)
                needed = largest_seed_region(partition, self.args.halo_rings)
                if needed > self.args.region_residues:
                    raise ValueError(
                        f'{full_name}: --region_residues {self.args.region_residues} is below '
                        f'the {needed} residues some seed group needs with '
                        f'--halo_rings {self.args.halo_rings}')


def frame0_ca(arr):
    """C-alpha (atom14 slot 1) of the trajectory's first frame, the one partitions are built on."""
    return np.asarray(arr[0, :, 1], dtype=np.float32)


def sample_region(partition, ca, args, generator=None):
    """(selection, region partition) for one Training region of a full-protein partition."""
    selection = sample_region_groups(partition, ca, region_residues=args.region_residues,
                                     halo_rings=args.halo_rings, generator=generator)
    return selection, region_subpartition(partition, selection)


REGION_AXIS = {'torsions': 1, 'trans': 1, 'rots': 1, 'torsion_mask': 0, 'seqres': 0,
               'mask': 0, 'chain_id': 0, 'residue_index': 0}


def pad_region(item, selection, region, width):
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

    out = {key: padded(value, REGION_AXIS[key]) if key in REGION_AXIS else value
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
        'halo_groups': len(selection.halo_groups),
        'residues': n,
        'loss_residues': int(region.loss_core_mask.sum()),
        'next_region_size': selection.next_region_size,
    }
    return out


def collate(items):
    """default_collate, keeping each item's sub-partition and region summary as lists."""
    keep = [key for key in ('partition', 'region') if key in items[0]]
    batch = default_collate([{k: v for k, v in item.items() if k not in keep} for item in items])
    for key in keep:
        batch[key] = [item[key] for item in items]
    return batch
