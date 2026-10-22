"""`no_frames` items carry per-residue atom37 masks."""
from types import SimpleNamespace

import numpy as np
from torch.utils.data import default_collate

from mdgen.dataset import MDGenDataset
from mdgen.residue_constants import atom_order, restype_order

T = 3


def test_no_frames_item_has_atom37_and_per_residue_mask(tmp_path):
    seq = 'GAW'
    (tmp_path / 'split.csv').write_text(f'name,seqres\nprot,{seq}\n')
    np.save(tmp_path / 'prot.npy',
            np.random.default_rng(0).normal(size=(T, len(seq), 14, 3)).astype(np.float32))
    args = SimpleNamespace(
        data_dir=str(tmp_path), suffix='', frame_interval=None, num_frames=T, overfit=False,
        overfit_peptide=None, overfit_frame=False, copy_frames=False, no_frames=True,
        atlas=False, training_region=False)

    item = MDGenDataset(args, tmp_path / 'split.csv')[0]
    assert item['atom37'].shape == (T, len(seq), 37, 3)
    assert item['seqres'].tolist() == [restype_order[c] for c in seq]
    assert item['mask'].shape == (len(seq), 37)
    assert item['mask'].sum(-1).tolist() == [4, 5, 14]  # heavy atoms of G, A, W
    assert (item['mask'][:, atom_order['CA']] == 1).all()

    # Shape that `prep_batch_no_frames` indexes as mask[:, None, :, :, None].
    batch = default_collate([item])
    assert batch['mask'][:, None, :, :, None].shape == (1, 1, len(seq), 37, 1)
