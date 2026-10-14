"""Compare FLRH tolerance runs using common bins and reference correlations."""
import argparse
import json
import pickle
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

parser = argparse.ArgumentParser()
parser.add_argument('root', type=Path)
args = parser.parse_args()
levels = ['baseline', 'tighter', 'tightest']
frames = {k: pd.read_csv(args.root / k / 'analysis/generated_frames.csv') for k in levels}
ref = pd.read_csv(args.root / 'baseline/analysis/reference_frames.csv')
rows = []
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fields = ['backbone_rmsd_A', 'end_CA_distance_A']
for field, ax in zip(fields, axes):
    bins = np.linspace(min([ref[field].min()] + [f[field].min() for f in frames.values()]),
                       max([ref[field].max()] + [f[field].max() for f in frames.values()]), 51)
    ax.hist(ref[field], bins=bins, density=True, histtype='step', label='MD', color='black')
    for level, f in frames.items():
        ax.hist(f[field], bins=bins, density=True, histtype='step', label=level)
    ax.set(xlabel=field, ylabel='Probability density')
    ax.legend()
fig.tight_layout()
fig.savefig(args.root / 'comparison_shapes.png', dpi=160)
fig.savefig(args.root / 'comparison_shapes.pdf')
plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for level, f in frames.items():
    d = args.root / level
    log = (d / 'inference.log').read_text()
    elapsed = re.findall(r'^([0-9]+\.[0-9]+)$', log, flags=re.MULTILINE)
    bonds = pd.read_csv(d / 'analysis/bonds.csv')
    peptide = bonds[bonds.atom_a.str.endswith('-C') & bonds.atom_b.str.endswith('-N')]
    result = pickle.loads((d / 'generated/out.pkl').read_bytes())['FLRH']
    row = {'setting': level, 'frames': len(f),
           'generation_seconds': float(elapsed[-1]) if elapsed else None,
           'peptide_bonds_outside_MD_p01_p99_fraction': float(peptide.generated_outside_reference_fraction.mean()),
           'close_contact_fraction': float((f.close_contact_pairs > 0).mean()),
           'rmsd_median_A': float(f.backbone_rmsd_A.median()),
           'tica_joint_js_distance': float(result['JSD']['TICA-0,1'])}
    for field in fields:
        bins = np.linspace(min([ref[field].min()] + [v[field].min() for v in frames.values()]),
                           max([ref[field].max()] + [v[field].max() for v in frames.values()]), 51)
        row[field + '_js_distance'] = float(jensenshannon(np.histogram(ref[field], bins)[0], np.histogram(f[field], bins)[0]))
    for key, ax in zip(['PHI', 'PSI'], axes):
        c = pd.read_csv(d / f'analysis/correlation_MDGen_{key}.csv')
        ax.plot(c.iloc[:, 0], c.iloc[:, 1], label=level)
        if level == 'baseline':
            r = pd.read_csv(d / f'analysis/correlation_MD_{key}.csv')
            ax.plot(r.iloc[:, 0], r.iloc[:, 1], label='MD', color='black')
        ax.set(xlabel='Lag (ns)', ylabel='Normalized correlation', title='LEU2 ' + key, xlim=(0, 10))
        ax.legend()
    rows.append(row)
fig.tight_layout()
fig.savefig(args.root / 'comparison_correlations.png', dpi=160)
fig.savefig(args.root / 'comparison_correlations.pdf')
pd.DataFrame(rows).to_csv(args.root / 'comparison.csv', index=False)
(args.root / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
print(pd.DataFrame(rows).to_string(index=False))
