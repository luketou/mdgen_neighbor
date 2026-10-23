"""ATLAS analysis-layout inputs through the existing preprocessing CLI (#43 T1)."""
import os
from pathlib import Path
import subprocess
import sys

import mdtraj as md
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from install_atlas_mirror import sha256, verify  # noqa: E402


def write_analysis(tmp_path, interval=100, seqres='GGG', xtc_residues=3):
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
    # xtc_residues < 3: the XTC belongs to a smaller system than the PDB
    trajectory.atom_slice(range(4 * xtc_residues)).save_xtc(str(source / 'prot_R1.xtc'))
    split = tmp_path / 'split.csv'
    split.write_text(f'name,seqres\nprot,{seqres}\n')
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


@pytest.mark.parametrize('source, message', [
    ({'interval': 200}, 'native frame interval differs'),
    ({'seqres': 'GGA'}, 'topology sequence differs'),
    ({'xtc_residues': 2}, 'ValueError'),  # mdtraj's atom-count message varies by version
])
def test_analysis_layout_rejects_bad_sources_without_output(tmp_path, source, message):
    split = write_analysis(tmp_path, **source)
    result = subprocess.run(prep_command(tmp_path, split), cwd=ROOT, capture_output=True,
                            text=True, env={**os.environ, 'PYTHONPATH': str(ROOT)})
    assert result.returncode != 0
    assert message in result.stderr
    assert not (tmp_path / 'out' / 'prot_R1.npy').exists()


def test_mirror_verify_rejects_size_and_hash_mismatch(tmp_path):
    path = tmp_path / 'prot_R1.xtc'
    path.write_bytes(b'mirror payload')
    entry = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    assert verify(path, entry)['sha256'] == entry['sha256']
    with pytest.raises(ValueError, match='Size mismatch'):
        verify(path, {**entry, 'bytes': entry['bytes'] + 1})
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        verify(path, {**entry, 'sha256': '0' * 64})
