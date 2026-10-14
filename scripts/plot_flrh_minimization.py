"""Matched before/after distributions for the minimization pilot."""
from pathlib import Path
import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('outputs/flrh_minimization_pilot')
trajs={label:md.load(str(out/(label+'.xtc')),top=str(out/(label+'.pdb'))) for label in ['generated_original','generated_k10000','generated_k1000','MD_control_original','MD_control_k10000']}
top=trajs['generated_original'].topology
pairs=np.array([[top.select(f'resid {r} and name C')[0],top.select(f'resid {r+1} and name N')[0]] for r in range(3)])
fig,axs=plt.subplots(1,3,figsize=(12,3.5))
for k,ax in enumerate(axs):
    for label in trajs:
        x=md.compute_distances(trajs[label],pairs,periodic=False)[:,k]*10
        ax.hist(x,bins=np.linspace(.8,1.95,47),density=True,histtype='step',label=label)
    ax.set(xlabel='Peptide C-N distance (angstrom)',ylabel='Density',title=f'Bond {k+1}')
axs[0].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'bond_distributions.png',dpi=160)
fig,axs=plt.subplots(1,2,figsize=(10,3.5))
for ax,compute,name in zip(axs,[md.compute_phi,md.compute_psi],['LEU2 phi','LEU2 psi']):
    for label in ['generated_original','generated_k10000','generated_k1000','MD_control_original']:
        atoms,angles=compute(trajs[label],periodic=False)
        idx=next(i for i,q in enumerate(atoms) if top.atom(int(q[1])).residue.index==1)
        ax.hist(np.rad2deg(angles[:,idx]),bins=np.linspace(-180,180,37),density=True,histtype='step',label=label)
    ax.set(xlabel='Degrees',ylabel='Density',title=name);ax.legend(fontsize=6)
fig.tight_layout();fig.savefig(out/'angle_distributions.png',dpi=160)
