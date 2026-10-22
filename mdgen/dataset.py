import json
import torch
from .full_length.region import sample_training_region
from .rigid_utils import Rigid
from .residue_constants import restype_order
import numpy as np
import pandas as pd
from .geometry import atom37_to_torsions, atom14_to_atom37, atom14_to_frames
       
class MDGenDataset(torch.utils.data.Dataset):
    def __init__(self, args, split, repeat=1, region_seed=None):
        super().__init__()
        self.df = pd.read_csv(split, index_col='name')
        self.args = args
        self.repeat = repeat
        self.region_seed = region_seed  # fixed Training regions, e.g. for validation
    def __len__(self):
        if self.args.overfit_peptide:
            return 1000
        return self.repeat * len(self.df)

    def __getitem__(self, idx):
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
            rng = np.random.RandomState(self.region_seed + idx)  # fix replica and window too

        if self.args.atlas:
            i = rng.randint(1, 4)
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
        # arr = np.copy(arr[frame_start:end]) * 10 # convert to angstroms
        arr = np.copy(arr[frame_start:end]).astype(np.float32) # / 10.0 # convert to nm
        if self.args.copy_frames:
            arr[1:] = arr[0]

        # arr should be in ANGSTROMS
        frames = atom14_to_frames(torch.from_numpy(arr))
        seqres = np.array([restype_order[c] for c in seqres])
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
                'mask': restype_atom37_mask[seqres], # (L,)
            }
        torsions, torsion_mask = atom37_to_torsions(atom37, aatype)
        
        torsion_mask = torsion_mask[0]

        if self.args.training_region:
            return self.training_region_item(idx, name, {
                'name': full_name,
                'frame_start': frame_start,
                'torsions': torsions,
                'torsion_mask': torsion_mask,
                'trans': frames._trans,
                'rots': frames._rots._rot_mats,
                'seqres': seqres,
                'mask': mask,
            })

        if self.args.atlas:
            if L > self.args.crop:
                start = np.random.randint(0, L - self.args.crop + 1)
                torsions = torsions[:,start:start+self.args.crop]
                frames = frames[:,start:start+self.args.crop]
                seqres = seqres[start:start+self.args.crop]
                mask = mask[start:start+self.args.crop]
                torsion_mask = torsion_mask[start:start+self.args.crop]
                
            
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

        return {
            'name': full_name,
            'frame_start': frame_start,
            'torsions': torsions,
            'torsion_mask': torsion_mask,
            'trans': frames._trans,
            'rots': frames._rots._rot_mats,
            'seqres': seqres,
            'mask': mask, # (L,)
        }

    def training_region_item(self, idx, name, item):
        """Read optional chain numbering from the split CSV, then gather a region."""
        L = len(item['seqres'])
        numbering = {'chain_id': np.zeros(L, dtype=np.int64), 'residue_index': np.arange(L)}
        for key in numbering:  # optional per-residue JSON lists in the split CSV
            if key in self.df and name in self.df.index:
                numbering[key] = np.array(json.loads(self.df[key][name]), dtype=np.int64)
                if len(numbering[key]) != L:
                    raise ValueError(f'{name}: {key} has {len(numbering[key])} entries, expected {L}')
        generator = None
        if self.region_seed is not None:
            generator = torch.Generator().manual_seed(self.region_seed + idx)
        return gather_training_region(item, numbering['chain_id'], numbering['residue_index'],
                                      self.args, generator)


PER_RESIDUE_AXIS = {'torsions': 1, 'trans': 1, 'rots': 1,
                    'torsion_mask': 0, 'seqres': 0, 'mask': 0}


def gather_training_region(item, chain_id, residue_index, args, generator=None):
    """Sample a Training region from item's first frame; gather it in residue order.

    Per-residue fields are padded to region_core + region_ghost; `core_mask`
    marks the residues that count toward the loss, `mask` every real residue,
    and `region` holds the center and diagnostics. Other fields pass through.
    """
    region = sample_training_region(
        item['trans'][0],
        mask=torch.as_tensor(item['mask']),
        chain_id=torch.as_tensor(chain_id),
        residue_index=torch.as_tensor(residue_index),
        core_count=args.region_core,
        ghost_capacity=args.region_ghost,
        contact_radius=args.spatial_contact_radius,
        generator=generator,
    )
    keep = torch.cat([region.core_indices, region.ghost_indices]).sort().values
    width = args.region_core + args.region_ghost
    pad = width - len(keep)

    def gather(value, axis):
        value = value[:, keep] if axis else value[keep]
        if isinstance(value, np.ndarray):
            return np.concatenate([value, np.zeros((pad, *value.shape[1:]), dtype=value.dtype)])
        shape = list(value.shape)
        shape[axis] = pad
        return torch.cat([value, value.new_zeros(shape)], axis)

    out = {key: gather(value, PER_RESIDUE_AXIS[key]) if key in PER_RESIDUE_AXIS else value
           for key, value in item.items()}
    out['rots'][:, len(keep):] = torch.eye(3)
    out['core_mask'] = np.zeros(width, dtype=np.float32)
    out['core_mask'][:len(keep)] = np.isin(keep.numpy(), region.core_indices.numpy())
    out['chain_id'] = gather(np.asarray(chain_id, dtype=np.int64), 0)
    out['residue_index'] = gather(np.asarray(residue_index, dtype=np.int64), 0)
    details = region.diagnostics
    out['region'] = {'center': region.center, **vars(details),
                     'spatial_truncated_count': details.spatial_truncated_count}
    return out
