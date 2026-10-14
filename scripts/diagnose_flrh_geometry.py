"""Locate FLRH geometry differences in reconstruction, export and rollouts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mdgen.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions, frames_torsions_to_atom14
from mdgen.residue_constants import restype_order, restype_name_to_atom14_names
from mdgen.utils import atom14_to_pdb

p = argparse.ArgumentParser()
p.add_argument('--generated', required=True)
p.add_argument('--outdir', required=True)
a = p.parse_args()
out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
gen = md.load(a.generated + '.xtc', top=a.generated + '.pdb')
top = gen.topology
ref = md.load('outputs/preprocessed_flrh/FLRH.xtc', top='outputs/preprocessed_flrh/FLRH.pdb')
seq = np.array([restype_order[x] for x in 'FLRH'])
slots = [(atom.residue.index, restype_name_to_atom14_names[atom.residue.name].index(atom.name)) for atom in top.atoms]
ri = [next(x.index for x in ref.topology.atoms if x.residue.index == atom.residue.index and x.name == atom.name) for atom in top.atoms]
ref = md.Trajectory(ref.xyz[:, ri], top, time=ref.time)
pairs = np.array([[top.select(f'resid {r} and name C')[0], top.select(f'resid {r+1} and name N')[0]] for r in range(3)])
def bonds(x):
    return np.linalg.norm(x[:, pairs[:, 0]] - x[:, pairs[:, 1]], axis=-1)
def xyz(arr):
    return np.stack([arr[:, r, k] for r, k in slots], axis=1)
def stats(x):
    return {'min_A': float(x.min()), 'median_A': float(np.median(x)), 'max_A': float(x.max())}
gb = bonds(gen.xyz * 10); rb = bonds(ref.xyz * 10)
lo, hi = np.percentile(rb, [1, 99], axis=0)
# Reuse the exact close-contact screen from the basic analysis.
frames = pd.read_csv(out / 'generated_frames.csv')
choices = {'shortest_bond': int(np.unravel_index(gb.argmin(), gb.shape)[0]),
           'longest_bond': int(np.unravel_index(gb.argmax(), gb.shape)[0]),
           'most_contacts': int(frames.close_contact_pairs.idxmax())}
normal = np.flatnonzero(((gb >= lo) & (gb <= hi)).all(axis=1) & (frames.close_contact_pairs.to_numpy() == 0))
choices['normal_comparison'] = int(normal[len(normal)//2]) if len(normal) else 0
clips = []
for label, f in choices.items():
    start, end = max(0, f-10), min(gen.n_frames, f+11)
    clip = gen[start:end]
    clip[0].save_pdb(str(out / (label + '.pdb')))
    clip.save_xtc(str(out / (label + '.xtc')))
    gen[f].save_pdb(str(out / (label + '_selected.pdb')))
    clips.append({'label': label, 'frame_zero_based': f, 'clip_start': start, 'selected_index_in_xtc': f-start,
                  'bond_lengths_A': gb[f].tolist(), 'close_contacts': int(frames.close_contact_pairs.iloc[f])})
(out / 'selected_frames.json').write_text(json.dumps(clips, indent=2))
rows = []
for r in range(10):
    sl = slice(r*1000, (r+1)*1000)
    row = {'rollout': r+1, 'contact_fraction': float((frames.close_contact_pairs.iloc[sl] > 0).mean()),
           'rmsd_median_A': float(frames.backbone_rmsd_A.iloc[sl].median()),
           'peptide_outside_reference_fraction': float(((gb[sl] < lo) | (gb[sl] > hi)).mean())}
    for k in range(3):
        for key, value in stats(gb[sl, k]).items(): row[f'bond_{k+1}_{key}'] = value
    rows.append(row)
pd.DataFrame(rows).to_csv(out / 'rollout_geometry.csv', index=False)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, field in zip(axs, ['contact_fraction', 'peptide_outside_reference_fraction', 'rmsd_median_A']):
    ax.plot(range(1, 11), [r[field] for r in rows], marker='o'); ax.set(xlabel='Rollout', ylabel=field)
fig.tight_layout(); fig.savefig(out / 'rollout_geometry.png', dpi=160); plt.close(fig)
fig, axs = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axs.flat, ['phi_LEU2_deg', 'psi_LEU2_deg', 'phi_ARG3_deg', 'psi_ARG3_deg']):
    ax.hist(frames[col].iloc[:1000], bins=np.linspace(-180,180,49), density=True, histtype='step', label='rollout 1')
    ax.hist(frames[col].iloc[9000:], bins=np.linspace(-180,180,49), density=True, histtype='step', label='rollout 10')
    ax.hist(pd.read_csv(out / 'reference_frames.csv')[col], bins=np.linspace(-180,180,49), density=True, histtype='step', label='MD')
    ax.set(title=col, xlabel='Degrees', ylabel='Density'); ax.legend()
fig.tight_layout(); fig.savefig(out / 'rollout_angles.png', dpi=160); plt.close(fig)
# Known MD -> actual rigid/torsion reconstruction path, sampled across the MD trajectory.
arr = np.load('data/4AA_data/FLRH_i100.npy')[::100].astype(np.float32)
t = torch.from_numpy(arr); aa = torch.from_numpy(seq)[None].expand(len(arr), -1)
f = atom14_to_frames(t)
a37 = torch.from_numpy(atom14_to_atom37(arr, seq[None])).float()
torsions, _ = atom37_to_torsions(a37, aa)
with torch.no_grad(): rebuilt = frames_torsions_to_atom14(f, torsions, aa).numpy()
x, y = xyz(arr), xyz(rebuilt)
reconstruction = {'sampled_frames': len(arr), 'atom_displacement_A': stats(np.linalg.norm(x-y, axis=-1)),
                  'input_peptide_bonds': stats(bonds(x)), 'rebuilt_peptide_bonds': stats(bonds(y)),
                  'peptide_bond_change_A': stats(np.abs(bonds(x)-bonds(y)))}
# Compare saved generated atom14 to final XTC using rotation-invariant bond lengths.
raw = np.load(str(Path(a.generated).parent / 'FLRH_atom14.npy'))
rawxyz = xyz(raw)
assert raw.shape[0] == gen.n_frames and np.isfinite(raw).all()
rawb = bonds(rawxyz)
# Direct export check on the four selected generated frames.
selected = np.array(list(choices.values()))
atom14_to_pdb(raw[selected], seq, str(out / 'export_check.pdb'))
exported = md.load(str(out / 'export_check.pdb'))
assert [(x.residue.index,x.name) for x in exported.topology.atoms] == [(x.residue.index,x.name) for x in top.atoms]
pdbb = bonds(exported.xyz * 10)
exported.save_xtc(str(out / 'export_check.xtc'))
readback = md.load(str(out / 'export_check.xtc'), top=str(out / 'export_check.pdb'))
export = {'raw_peptide_bonds': stats(rawb), 'final_xtc_peptide_bonds': stats(gb),
          'raw_to_final_xtc_bond_difference_A': stats(np.abs(rawb-gb)),
          'raw_to_pdb_bond_difference_A': stats(np.abs(rawb[selected]-pdbb)),
          'pdb_to_xtc_bond_difference_A': stats(np.abs(pdbb-bonds(readback.xyz*10)))}
report = {'reconstruction': reconstruction, 'export': export, 'selected_frames': clips,
          'notes': ['Reconstruction uses ideal residue geometry and is not an exact inverse of MD coordinates.',
                    'Peptide bond lengths are measured between separate residue rigid frames.',
                    'Reference percentiles are empirical comparison ranges, not physical validity limits.',
                    'Raw atom14 is after reconstruction, before serialization; latent tensors were not captured.']}
(out / 'diagnosis.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
