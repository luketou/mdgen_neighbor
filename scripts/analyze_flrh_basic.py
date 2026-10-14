"""Compare the preprocessed FLRH reference and one generated trajectory on CPU."""
import argparse
import json
from pathlib import Path

import mdtraj as md
import numpy as np
import pandas as pd


def describe(values):
    return dict(zip(('min', 'median', 'max'), map(float, (
        np.min(values), np.median(values), np.max(values)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='outputs/preprocessed_flrh/FLRH')
    parser.add_argument('--generated', default='outputs/demo_flrh_47009/FLRH')
    parser.add_argument('--outdir', default='outputs/analysis_flrh_47009')
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ref = md.load(args.reference + '.xtc', top=args.reference + '.pdb')
    gen = md.load(args.generated + '.xtc', top=args.generated + '.pdb')
    identity = lambda t: [(a.residue.index, a.residue.name, a.name)
                          for a in t.topology.atoms]
    ref_keys, gen_keys = identity(ref), identity(gen)
    if len(set(ref_keys)) != len(ref_keys) or set(ref_keys) != set(gen_keys):
        raise ValueError('Reference and generated atom identities must match uniquely')
    # The source PDB and MDGen output use different atom orders.
    lookup = {key: i for i, key in enumerate(ref_keys)}
    ref = md.Trajectory(ref.xyz[:, [lookup[key] for key in gen_keys]],
                        gen.topology, time=ref.time)
    top = gen.topology
    backbone = top.select('backbone')
    ca = top.select('name CA')
    bonds = sorted({tuple(sorted((a.index, b.index))) for a, b in top.bonds})
    if not bonds:
        raise ValueError('No bonds inferred from topology')
    neighbors = [set() for _ in top.atoms]
    for a, b in bonds:
        neighbors[a].add(b)
        neighbors[b].add(a)
    # Exclude directly bonded pairs and pairs separated by two bonds (1-3).
    pairs = [(a, b) for a in range(top.n_atoms) for b in range(a + 1, top.n_atoms)
             if b not in neighbors[a] and not (neighbors[a] & neighbors[b])]
    # Heuristic heavy-atom overlap screen; not a force-field energy test.
    radii = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80}
    limits = np.array([0.6 * (radii[top.atom(a).element.symbol]
                             + radii[top.atom(b).element.symbol]) for a, b in pairs])
    summary = {'notes': [
        'Distances in angstrom; RMSD uses backbone fitted to reference frame 0.',
        'Frame index is used; XTC output timestamps are not used as physical time.',
        'Clash screen: distance < 0.6*(vdW radius sum), excluding 1-2 and 1-3 pairs.',
        'Reference bond 1st-99th percentiles are comparison ranges, not physical limits.',
        'One generated trajectory is an initial check, not a scientific validation.']}
    ref_bonds = md.compute_distances(ref, bonds, periodic=False) * 10
    lower, upper = np.percentile(ref_bonds, [1, 99], axis=0)
    bond_table = pd.DataFrame({
        'atom_a': [str(top.atom(a)) for a, _ in bonds],
        'atom_b': [str(top.atom(b)) for _, b in bonds],
        'reference_p01_A': lower, 'reference_p99_A': upper,
        'reference_median_A': np.median(ref_bonds, axis=0)})
    for label, traj in [('reference', ref), ('generated', gen)]:
        rmsd = md.rmsd(traj, ref, 0, atom_indices=backbone) * 10
        end_distance = md.compute_distances(traj, [[ca[0], ca[-1]]], periodic=False)[:, 0] * 10
        lengths = md.compute_distances(traj, bonds, periodic=False) * 10
        distances = md.compute_distances(traj, pairs, periodic=False) * 10
        overlaps = np.sum(distances < limits[None, :], axis=1)
        table = pd.DataFrame({'frame': np.arange(traj.n_frames),
                              'backbone_rmsd_A': rmsd, 'end_CA_distance_A': end_distance,
                              'close_contact_pairs': overlaps})
        for kind, compute in [('phi', md.compute_phi), ('psi', md.compute_psi)]:
            atoms, angles = compute(traj, periodic=False)
            for k, quartet in enumerate(atoms):
                residue = top.atom(int(quartet[1])).residue
                table[f'{kind}_{residue.name}{residue.index + 1}_deg'] = np.rad2deg(angles[:, k])
        table.to_csv(outdir / f'{label}_frames.csv', index=False)
        bond_table[f'{label}_min_A'] = lengths.min(axis=0)
        bond_table[f'{label}_max_A'] = lengths.max(axis=0)
        bond_table[f'{label}_outside_reference_fraction'] = np.mean(
            (lengths < lower) | (lengths > upper), axis=0)
        summary[label] = {'frames': traj.n_frames, 'atoms': traj.n_atoms,
                          'backbone_rmsd_A': describe(rmsd),
                          'end_CA_distance_A': describe(end_distance),
                          'frames_with_close_contacts_fraction': float(np.mean(overlaps > 0)),
                          'close_contact_pairs_per_frame': describe(overlaps)}
    bond_table.to_csv(outdir / 'bonds.csv', index=False)
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
