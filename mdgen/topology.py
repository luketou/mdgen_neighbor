"""Per-residue chain numbering read from an MD topology."""
import numpy as np


def chain_numbering(topology):
    """Return `chain_id` (topology chain index) and `residue_index` (PDB resSeq).

    Numbering gaps inside a chain are kept, so residues across a gap or in
    different chains are never treated as covalent neighbours.
    """
    residues = list(topology.residues)
    chain_id = np.array([residue.chain.index for residue in residues], dtype=np.int64)
    residue_index = np.array([residue.resSeq for residue in residues], dtype=np.int64)
    return chain_id, residue_index


def save_chain_numbering(path, topology):
    chain_id, residue_index = chain_numbering(topology)
    np.savez(path, chain_id=chain_id, residue_index=residue_index)
