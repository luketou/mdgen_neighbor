"""Exploratory FLRH comparison plots and checks at rollout boundaries."""
import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mdtraj as md
import numpy as np
import pandas as pd

from mdgen.analysis import get_featurized_traj, get_tica


parser = argparse.ArgumentParser()
parser.add_argument('--outdir', default='outputs/analysis_flrh_10rollouts_47010')
parser.add_argument('--generated', default='outputs/demo_flrh_10rollouts_47010/FLRH')
parser.add_argument('--pyemma', default='outputs/pyemma_flrh_10rollouts_47010/out.pkl')
args = parser.parse_args()
OUT = Path(args.outdir)
GEN = args.generated
REF = 'outputs/preprocessed_flrh/FLRH'
COLORS = ['#276FBF', '#D47732']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / (name + '.png'), dpi=160)
    fig.savefig(OUT / (name + '.pdf'))
    plt.close(fig)


def main():
    ref = pd.read_csv(OUT / 'reference_frames.csv')
    gen = pd.read_csv(OUT / 'generated_frames.csv')
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, field, title in zip(axs, ['backbone_rmsd_A', 'end_CA_distance_A'],
                               ['Backbone change from reference frame 0', 'Bending / extension']):
        bins = np.linspace(min(ref[field].min(), gen[field].min()),
                           max(ref[field].max(), gen[field].max()), 50)
        for df, label, color in zip([ref, gen], ['MD', 'MDGen'], COLORS):
            ax.hist(df[field], bins=bins, density=True, histtype='step',
                    linewidth=1.8, label=label, color=color)
        ax.set(xlabel=field.replace('_', ' '), ylabel='Probability density', title=title)
        ax.legend()
    save(fig, '01_shape_distributions')

    fig, axs = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    edges = np.linspace(-180, 180, 49)
    grids = []
    for residue in ['LEU2', 'ARG3']:
        for df in [ref, gen]:
            h, _, _ = np.histogram2d(df[f'phi_{residue}_deg'], df[f'psi_{residue}_deg'],
                                    bins=[edges, edges])
            grids.append(h / h.sum())
    positives = np.concatenate([h[h > 0] for h in grids])
    norm = LogNorm(vmin=positives.min(), vmax=positives.max())
    for ax, h, title in zip(axs.flat, grids, ['MD: LEU2', 'MDGen: LEU2', 'MD: ARG3', 'MDGen: ARG3']):
        im = ax.pcolormesh(edges, edges, np.ma.masked_equal(h.T, 0), norm=norm, cmap='viridis')
        ax.set(title=title, xlabel='phi (degrees)', ylabel='psi (degrees)',
               xticks=[-180, -90, 0, 90, 180], yticks=[-180, -90, 0, 90, 180])
    fig.tight_layout(rect=[0, 0, .87, 1])
    cax = fig.add_axes([.89, .15, .025, .7])
    fig.colorbar(im, cax=cax, label='Probability per angular bin (shared log scale)')
    fig.savefig(OUT / '02_backbone_angles.png', dpi=160)
    fig.savefig(OUT / '02_backbone_angles.pdf')
    plt.close(fig)

    results = pickle.loads(Path(args.pyemma).read_bytes())['FLRH']
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, field in zip(axs, ['PHI 0 LEU 2', 'PSI 0 LEU 2']):
        for key, dt, label, color in [('md_decorrelation', .1, 'MD', COLORS[0]),
                                      ('our_decorrelation', 10., 'MDGen', COLORS[1])]:
            y = np.asarray(results[key][field], dtype=float)
            time = np.arange(len(y)) * dt / 1000
            ax.plot(time, y, label=label, color=color)
            pd.DataFrame({'lag_ns': time, 'correlation': y}).to_csv(
                OUT / f"correlation_{label}_{field.split()[0]}.csv", index=False)
        ax.set(title=field, xlabel='Lag time (ns)', ylabel='Circular-angle correlation', xlim=(0, 10))
        ax.legend()
    save(fig, '03_time_correlations')

    fig, ax = plt.subplots(figsize=(9, 3.5))
    x = np.arange(10)
    for offset, key, label, color in [(-.2, 'ref_metastable_probs', 'MD', COLORS[0]),
                                     (.2, 'traj_metastable_probs', 'MDGen', COLORS[1])]:
        ax.bar(x + offset, results[key], .4, label=label, color=color)
    ax.set(xlabel='State index (same MD-based assignment)', ylabel='Fraction of frames', xticks=x)
    ax.legend()
    save(fig, '04_state_populations')

    rt = md.load(REF + '.xtc', top=REF + '.pdb')
    gt = md.load(GEN + '.xtc', top=GEN + '.pdb')
    keys = lambda t: [(a.residue.index, a.name) for a in t.topology.atoms]
    lookup = {k: i for i, k in enumerate(keys(rt))}
    rt = md.Trajectory(rt.xyz[:, [lookup[k] for k in keys(gt)]], gt.topology)
    peptide_bonds = [(a.index, b.index) for a, b in gt.topology.bonds
                     if a.residue.index != b.residue.index]
    fig, axs = plt.subplots(1, len(peptide_bonds), figsize=(12, 3.5))
    lengths_ref = md.compute_distances(rt, peptide_bonds, periodic=False) * 10
    lengths_gen = md.compute_distances(gt, peptide_bonds, periodic=False) * 10
    for k, (ax, (a, b)) in enumerate(zip(axs, peptide_bonds)):
        bins = np.linspace(min(lengths_ref[:, k].min(), lengths_gen[:, k].min()),
                           max(lengths_ref[:, k].max(), lengths_gen[:, k].max()), 60)
        for values, label, color in [(lengths_ref, 'MD', COLORS[0]), (lengths_gen, 'MDGen', COLORS[1])]:
            ax.hist(values[:, k], bins=bins, density=True, histtype='step', linewidth=1.5,
                    label=label, color=color)
        ax.set(title=f'{gt.topology.atom(a)} -- {gt.topology.atom(b)}',
               xlabel='Peptide bond length (angstrom)', ylabel='Probability density')
        ax.legend()
    save(fig, '05_peptide_bonds')

    # Fit every neighboring pair independently; remove global translation/rotation.
    xyz = gt.xyz[:, gt.topology.select('backbone')].astype(np.float64) * 10
    xyz -= xyz.mean(axis=1, keepdims=True)
    covariance = np.einsum('tni,tnj->tij', xyz[:-1], xyz[1:])
    u, s, vh = np.linalg.svd(covariance)
    trace = s[:, :2].sum(axis=1) + np.linalg.det(u @ vh) * s[:, 2]
    error = ((xyz[:-1] ** 2).sum(axis=(1, 2)) + (xyz[1:] ** 2).sum(axis=(1, 2)) - 2 * trace)
    step_rmsd = np.sqrt(np.maximum(error / xyz.shape[1], 0))
    boundary = np.arange(1, len(gt)) % 1000 == 0
    table = pd.DataFrame({'frame_after_step': np.arange(1, len(gt)),
                          'neighbor_backbone_rmsd_A': step_rmsd, 'rollout_boundary': boundary})
    table.to_csv(OUT / 'neighbor_steps.csv', index=False)
    table[boundary].to_csv(OUT / 'rollout_boundaries.csv', index=False)
    stats = {'within_segment_median_A': float(np.median(step_rmsd[~boundary])),
             'within_segment_p99_A': float(np.percentile(step_rmsd[~boundary], 99)),
             'boundary_steps_A': step_rmsd[boundary].tolist(),
             'boundary_frames_zero_based': table.loc[boundary, 'frame_after_step'].tolist()}
    (OUT / 'boundary_summary.json').write_text(json.dumps(stats, indent=2) + '\n')
    fig, axs = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    t = gen.frame.to_numpy() * .01
    axs[0].plot(t, gen.backbone_rmsd_A, lw=.6, color=COLORS[1])
    axs[0].set(ylabel='Backbone RMSD (angstrom)', title='Long rollout: structure and close-contact screen')
    axs[1].plot(t, gen.close_contact_pairs, lw=.6, color=COLORS[1])
    axs[1].set(xlabel='Nominal time (ns; 10 ps/frame)', ylabel='Flagged close-contact pairs')
    for ax in axs:
        for frame in stats['boundary_frames_zero_based']:
            ax.axvline(frame * .01, color='gray', ls=':', lw=.7)
    save(fig, '06_long_rollout_geometry')

    # Fit to raw MD exactly as in the original analysis, then project both trajectories.
    _, raw_features = get_featurized_traj('data/4AA_sims/FLRH/FLRH', sidechains=True, cossin=True)
    _, generated_features = get_featurized_traj(GEN, sidechains=True, cossin=True)
    tica, ref_tica = get_tica(raw_features)
    gen_tica = tica.transform(generated_features)
    xedges = np.linspace(min(ref_tica[:, 0].min(), gen_tica[:, 0].min()),
                         max(ref_tica[:, 0].max(), gen_tica[:, 0].max()), 61)
    yedges = np.linspace(min(ref_tica[:, 1].min(), gen_tica[:, 1].min()),
                         max(ref_tica[:, 1].max(), gen_tica[:, 1].max()), 61)
    grids = [np.histogram2d(v[:, 0], v[:, 1], bins=(xedges, yedges))[0] for v in [ref_tica, gen_tica]]
    grids = [h / h.sum() for h in grids]
    positive = np.concatenate([h[h > 0] for h in grids])
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True, layout='constrained')
    norm = LogNorm(vmin=positive.min(), vmax=positive.max())
    for ax, h, title in zip(axs, grids, ['MD', 'MDGen']):
        im = ax.pcolormesh(xedges, yedges, np.ma.masked_equal(h.T, 0), norm=norm, cmap='viridis')
        ax.set(title=title, xlabel='TICA 1', ylabel='TICA 2')
    fig.colorbar(im, ax=axs, label='Probability per bin (shared log scale)')
    fig.savefig(OUT / '07_tica_shared_scale.png', dpi=160)
    fig.savefig(OUT / '07_tica_shared_scale.pdf')
    plt.close(fig)
    print('Saved comparison figures and boundary checks to', OUT)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
