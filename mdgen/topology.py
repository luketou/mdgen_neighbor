"""Per-residue chain numbering read from an MD topology."""
from collections import Counter
import os

import numpy as np


def chain_numbering(topology):
    """Return `chain_id` (topology chain index) and `residue_index` (PDB resSeq).

    Numbering gaps inside a chain are kept, so residues across a gap or in
    different chains are never treated as covalent neighbours. Repeated resSeq
    within a chain (e.g. insertion codes) raise here rather than mid-training.
    """
    residues = list(topology.residues)
    chain_id = np.array([residue.chain.index for residue in residues], dtype=np.int64)
    residue_index = np.array([residue.resSeq for residue in residues], dtype=np.int64)
    counts = Counter(zip(chain_id.tolist(), residue_index.tolist()))
    repeated = sorted(pair for pair, count in counts.items() if count > 1)
    if repeated:
        raise ValueError('repeated residue numbers (insertion codes?): ' + ', '.join(
            f'chain {chain} residue {number}' for chain, number in repeated))
    return chain_id, residue_index


def default_numbering(length):
    """Single chain numbered by position: what the model assumes without chain files."""
    return {'chain_id': np.zeros(length, dtype=np.int64), 'residue_index': np.arange(length)}


def load_chain_numbering(path, length):
    """Numbering from a chain file written by scripts/prep_sims.py, or None if absent."""
    if not os.path.exists(path):
        return None
    with np.load(path) as saved:
        numbering = {key: saved[key].astype(np.int64) for key in ('chain_id', 'residue_index')}
    if any(len(value) != length for value in numbering.values()):
        raise ValueError(f'{path}: expected {length} residues')
    return numbering
