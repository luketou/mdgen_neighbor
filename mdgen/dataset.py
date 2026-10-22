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

        if self.args.atlas:
            i = np.random.randint(1, 4)
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
            frame_start = np.random.randint(0, max_start + 1)
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
        """Gather one Training region in residue order and pad it to K + G.

        `core_mask` marks the residues that count toward the loss; `mask` still
        marks every real residue (core and ghost).
        """
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
        region = sample_training_region(
            item['trans'][0],
            mask=torch.from_numpy(item['mask']),
            chain_id=torch.from_numpy(numbering['chain_id']),
            residue_index=torch.from_numpy(numbering['residue_index']),
            core_count=self.args.region_core,
            ghost_capacity=self.args.region_ghost,
            contact_radius=self.args.spatial_contact_radius,
            generator=generator,
        )
        keep = torch.cat([region.core_indices, region.ghost_indices]).sort().values
        width = self.args.region_core + self.args.region_ghost
        pad = width - len(keep)

        def gather(value, axis):
            value = value[:, keep] if axis else value[keep]
            if isinstance(value, np.ndarray):
                return np.concatenate([value, np.zeros((pad, *value.shape[1:]), dtype=value.dtype)])
            shape = list(value.shape)
            shape[axis] = pad
            return torch.cat([value, value.new_zeros(shape)], axis)

        rots = gather(item['rots'], 1)
        rots[:, len(keep):] = torch.eye(3)
        core_mask = np.zeros(width, dtype=np.float32)
        core_mask[:len(keep)] = np.isin(keep.numpy(), region.core_indices.numpy())
        return {
            'name': item['name'],
            'frame_start': item['frame_start'],
            'torsions': gather(item['torsions'], 1),
            'torsion_mask': gather(item['torsion_mask'], 0),
            'trans': gather(item['trans'], 1),
            'rots': rots,
            'seqres': gather(item['seqres'], 0),
            'mask': gather(item['mask'], 0),
            'core_mask': core_mask,
            'chain_id': gather(numbering['chain_id'], 0),
            'residue_index': gather(numbering['residue_index'], 0),
        }
