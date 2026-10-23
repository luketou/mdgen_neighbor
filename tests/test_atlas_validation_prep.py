"""ATLAS analysis-layout inputs through the existing preprocessing CLI (#43 T1)."""
import os
from pathlib import Path
import subprocess
import sys

import mdtraj as md
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_analysis(tmp_path, interval=100):
    top = md.Topology()
    chain = top.add_chain()
    for number in (1, 2, 4):
        residue = top.add_residue('GLY', chain, resSeq=number)
        for atom, element in [('N', 'N'), ('CA', 'C'), ('C', 'C'), ('O', 'O')]:
            top.add_atom(atom, md.element.get_by_symbol(element), residue)
    xyz = np.random.default_rng(137).normal(size=(9, 12, 3)).astype(np.float32)
    trajectory = md.Trajectory(xyz, top, time=np.arange(9) * interval)
    source = tmp_path / 'sims' / 'prot'
    source.mkdir(parents=True)
    trajectory.save_pdb(str(source / 'prot.pdb'))
    trajectory.save_xtc(str(source / 'prot_R1.xtc'))
    split = tmp_path / 'split.csv'
    split.write_text('name,seqres\nprot,GGG\n')
    return split


def prep_command(tmp_path, split):
    return [sys.executable, 'scripts/prep_sims.py', '--split', str(split),
            '--atlas_dir', str(tmp_path / 'sims'), '--outdir', str(tmp_path / 'out'),
            '--atlas', '--replicates', '1', '--atlas_layout', 'analysis',
            '--expected_interval_ps', '100', '--stride', '4', '--dtype', 'float32']


def test_analysis_layout_preserves_chain_gaps_and_samples_once(tmp_path):
    split = write_analysis(tmp_path)
    result = subprocess.run(prep_command(tmp_path, split), cwd=ROOT, capture_output=True,
                            text=True, env={**os.environ, 'PYTHONPATH': str(ROOT)})
    assert result.returncode == 0, result.stderr
    atoms = np.load(tmp_path / 'out' / 'prot_R1.npy')
    chain = np.load(tmp_path / 'out' / 'prot_R1_chain.npz')
    assert atoms.shape == (3, 3, 14, 3)  # source frames 0, 4, 8; not a second stride
    assert atoms.dtype == np.float32
    assert chain['residue_index'].tolist() == [1, 2, 4]
    assert chain['chain_id'].tolist() == [0, 0, 0]
