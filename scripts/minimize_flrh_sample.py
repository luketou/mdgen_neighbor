"""Restrained minimization pilot; GBn2/HIE assumptions, not original explicit MD."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import mdtraj as md
import openmm as mm
from openmm import app, unit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('--controls_only',action='store_true')
args=parser.parse_args()
OUT = Path('outputs/flrh_minimization_pilot'); OUT.mkdir(parents=True, exist_ok=True)
GEN = 'outputs/flrh_geometry_47039/generated/FLRH'
gen = md.load(GEN+'.xtc', top=GEN+'.pdb')
ref = md.load('outputs/preprocessed_flrh/FLRH.xtc', top='outputs/preprocessed_flrh/FLRH.pdb')
ri = [next(a.index for a in ref.topology.atoms if a.residue.index==b.residue.index and a.name==b.name) for b in gen.topology.atoms]
ref = md.Trajectory(ref.xyz[:,ri], gen.topology, time=ref.time)
top = gen.topology
pairs = np.array([[top.select(f'resid {r} and name C')[0],top.select(f'resid {r+1} and name N')[0]] for r in range(3)])
def bl(x): return np.linalg.norm(x[:,pairs[:,0]]-x[:,pairs[:,1]],axis=-1)*10
reference_lengths=bl(ref.xyz)
lo,hi=np.percentile(reference_lengths,[1,99],axis=0)
gb=bl(gen.xyz)
f=pd.read_csv('outputs/flrh_geometry_47039/analysis/generated_frames.csv')
uniform=np.linspace(0,len(gen)-1,80,dtype=int)
score=np.maximum(lo-gb,gb-hi).max(axis=1)+f.close_contact_pairs.to_numpy()*.05
extreme=[int(i) for i in np.argsort(score)[::-1] if i not in set(uniform)][:20]
selection=pd.DataFrame({'frame':np.r_[uniform,extreme],'group':['uniform']*80+['extreme']*20})
selection.to_csv(OUT/'selection.csv',index=False)
mdindices=np.linspace(0,len(ref)-1,20,dtype=int)
# Use chemically complete source topology; synthesize OXT equally for all samples.
source=app.PDBFile('data/4AA_sims/FLRH/FLRH.pdb')
source_atoms=list(source.topology.atoms())
lookup={(a.residue.index,a.name):a.index for a in top.atoms}
map_common=[next(a.index for a in source_atoms if (a.residue.index,a.name)==(b.residue.index,b.name)) for b in top.atoms]
c=top.select('resid 3 and name C')[0]; ca=top.select('resid 3 and name CA')[0]; o=top.select('resid 3 and name O')[0]
oxt=next(a.index for a in source_atoms if a.name=='OXT')
ff=app.ForceField('amber14-all.xml','implicit/gbn2.xml')
platform=mm.Platform.getPlatformByName('CPU')
properties={'Threads':'1'}
# Close-contact rule identical to prior analysis.
bondset={tuple(sorted((a.index,b.index))) for a,b in top.bonds}
adj={i:set() for i in range(top.n_atoms)}
for i,j in bondset: adj[i].add(j);adj[j].add(i)
cp=np.array([(i,j) for i in range(top.n_atoms) for j in range(i+1,top.n_atoms) if (i,j) not in bondset and not(adj[i]&adj[j])])
radii={'C':1.7,'N':1.55,'O':1.52,'S':1.8}
limits=np.array([.6*(radii[top.atom(i).element.symbol]+radii[top.atom(j).element.symbol]) for i,j in cp])
def contacts(x): return (np.linalg.norm(x[:,cp[:,0]]-x[:,cp[:,1]],axis=-1)*10<limits).sum(axis=1)
allrows=pd.read_csv(OUT/'per_frame.csv').query("source == 'generated'").to_dict('records') if args.controls_only else []
for label,traj,indices,groups in [('generated',gen,selection.frame.to_numpy(),selection.group.to_numpy()),('MD_control',ref,mdindices,['MD_control']*20)]:
    if args.controls_only and label=='generated': continue
    original=traj[indices]
    if label=='MD_control': assert not contacts(original.xyz).any(), 'MD reference atom ordering/contact screen mismatch'
    original[0].save_pdb(str(OUT/f'{label}_original.pdb')); original.save_xtc(str(OUT/f'{label}_original.xtc'))
    corrected={1000:[],10000:[]}
    for n,(index,group) in enumerate(zip(indices,groups)):
        x=traj.xyz[index].astype(float)
        complete=np.zeros((len(source_atoms),3));complete[map_common]=x
        axis=x[c]-x[ca];axis/=np.linalg.norm(axis)
        v=x[o]-x[c];complete[oxt]=x[c]+2*axis*np.dot(v,axis)-v
        random.seed(137+n)
        model=app.Modeller(source.topology,complete*unit.nanometer)
        model.addHydrogens(ff,pH=7.0,variants=[None,None,None,'HIE'],platform=platform)
        atoms=list(model.topology.atoms())
        common=[next(a.index for a in atoms if (a.residue.index,a.name)==(b.residue.index,b.name)) for b in top.atoms]
        initial=model.positions.value_in_unit(unit.nanometer)
        system=ff.createSystem(model.topology,nonbondedMethod=app.NoCutoff,constraints=None)
        restraint=mm.CustomExternalForce('0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
        restraint.addGlobalParameter('k',1000)
        for name in ['x0','y0','z0']:restraint.addPerParticleParameter(name)
        for atom in atoms:
            if atom.element!=app.element.hydrogen: restraint.addParticle(atom.index,initial[atom.index])
        system.addForce(restraint)
        integrator=mm.VerletIntegrator(.001*unit.picoseconds)
        context=mm.Context(system,integrator,platform,properties)
        for k in corrected:
            context.setPositions(model.positions);context.setParameter('k',k)
            before=context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            mm.LocalEnergyMinimizer.minimize(context,10,1000)
            state=context.getState(getPositions=True,getEnergy=True,getForces=True)
            pos=state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)[common]
            assert np.isfinite(pos).all()
            corrected[k].append(pos)
            forces=state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
            allrows.append({'source':label,'frame':int(index),'group':group,'k_kJ_mol_nm2':k,
                            'energy_before_kJ_mol':before,'energy_after_kJ_mol':state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                            'rms_force_kJ_mol_nm':float(np.sqrt(np.mean(np.sum(forces**2,axis=1))))})
        del context,integrator
        print(label,n+1,'/',len(indices),flush=True)
    for k,positions in corrected.items():
        after=md.Trajectory(np.array(positions),top)
        after[0].save_pdb(str(OUT/f'{label}_k{k}.pdb'));after.save_xtc(str(OUT/f'{label}_k{k}.xtc'))
        beforeb=bl(original.xyz);afterb=bl(after.xyz)
        for n in range(len(indices)):
            row=next(r for r in allrows if r['source']==label and r['frame']==int(indices[n]) and r['k_kJ_mol_nm2']==k)
            row.update({'rmsd_aligned_A':float(md.rmsd(after[n],original[n])[0]*10),
                        'displacement_rms_A':float(np.sqrt(np.mean(np.sum((after.xyz[n]-original.xyz[n])**2,axis=1)))*10),
                        'contacts_before':int(contacts(original.xyz[n:n+1])[0]),'contacts_after':int(contacts(after.xyz[n:n+1])[0]),
                        'bond_min_before_A':float(beforeb[n].min()),'bond_max_before_A':float(beforeb[n].max()),
                        'bond_min_after_A':float(afterb[n].min()),'bond_max_after_A':float(afterb[n].max()),
                        'bonds_outside_before':float(((beforeb[n]<lo)|(beforeb[n]>hi)).mean()),
                        'bonds_outside_after':float(((afterb[n]<lo)|(afterb[n]>hi)).mean())})
        for kind,compute in [('phi',md.compute_phi),('psi',md.compute_psi)]:
            _,b=compute(original,periodic=False);_,a=compute(after,periodic=False)
            diff=np.rad2deg(np.abs(np.angle(np.exp(1j*(a-b)))))
            for n in range(len(indices)):
                row=next(r for r in allrows if r['source']==label and r['frame']==int(indices[n]) and r['k_kJ_mol_nm2']==k)
                row[kind+'_mean_change_deg']=float(diff[n].mean())
        pd.DataFrame(allrows).to_csv(OUT/'per_frame.csv',index=False)
df=pd.DataFrame(allrows)
summary=df.groupby(['source','group','k_kJ_mol_nm2']).agg(n=('frame','size'),rmsd_median_A=('rmsd_aligned_A','median'),contacts_before_fraction=('contacts_before',lambda x:(x>0).mean()),contacts_after_fraction=('contacts_after',lambda x:(x>0).mean()),bond_outside_before=('bonds_outside_before','mean'),bond_outside_after=('bonds_outside_after','mean'),phi_change_median_deg=('phi_mean_change_deg','median'),psi_change_median_deg=('psi_mean_change_deg','median')).reset_index()
summary.to_csv(OUT/'summary.csv',index=False)
fig,axs=plt.subplots(1,3,figsize=(12,4))
for ax,field in zip(axs,['rmsd_aligned_A','phi_mean_change_deg','psi_mean_change_deg']):
    for source,color in [('generated','tab:blue'),('MD_control','tab:orange')]:
        for k,style in [(1000,'-'),(10000,'--')]:
            values=np.sort(df[(df.source==source)&(df.k_kJ_mol_nm2==k)][field])
            ax.plot(values,np.arange(1,len(values)+1)/len(values),style,color=color,label=f'{source} k={k}')
    ax.set(xlabel=field,ylabel='Cumulative fraction');ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(OUT/'structure_changes.png',dpi=160)
(OUT/'settings.json').write_text(json.dumps({'openmm':mm.__version__,'forcefield':['amber14-all.xml','implicit/gbn2.xml'],'histidine':'HIE assumed','pH':7,'termini':'free charged N/C; OXT synthesized identically for all samples','restraint_k_kJ_mol_nm2':[1000,10000],'minimizer_tolerance_kJ_mol_nm':10,'max_iterations':1000,'note':'Geometry pilot, not explicit-solvent MD reproduction; sampled frames are not a continuous trajectory.'},indent=2))
print(summary.to_string(index=False),flush=True)
