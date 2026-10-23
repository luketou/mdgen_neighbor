import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='splits/atlas.csv')
parser.add_argument('--atlas_dir', '--sim_dir', dest='atlas_dir', type=str, default='/data/cb/scratch/datasets/atlas')  # data-preprocess-fixes
parser.add_argument('--outdir', type=str, default='./data_atlas')
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--atlas', action='store_true')
parser.add_argument('--atlas_layout', choices=('protein', 'analysis'), default='protein',
                    help='protein: *_prod_Ri_fit.xtc; analysis: *_Ri.xtc (100 ps mirror).')
parser.add_argument('--expected_interval_ps', type=float, default=None,
                    help='Reject trajectories whose native time spacing differs, before striding.')
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--replicates', type=int, nargs='+', default=[1, 2, 3],
                    help='replicate numbers R<i> to convert in --atlas mode')
# npy-dtype-option:start
parser.add_argument(
    '--dtype',
    choices=('float16', 'float32'),
    default='float16',
)
# npy-dtype-option:end
args = parser.parse_args()
if args.stride < 1:
    parser.error('--stride must be positive')
if args.expected_interval_ps is not None and args.expected_interval_ps <= 0:
    parser.error('--expected_interval_ps must be positive')

import mdtraj, os, tqdm
import pandas as pd 
from multiprocessing import Pool
import numpy as np
from mdgen import residue_constants as rc
from mdgen.topology import chain_numbering

os.makedirs(args.outdir, exist_ok=True)

df = pd.read_csv(args.split, index_col='name')
names = df.index

def main():
    jobs = []
    for name in names:
        stem = f'{args.outdir}/{name}{args.suffix}'  # outputs are complete only with both files
        if os.path.exists(f'{stem}.npy') and os.path.exists(f'{stem}_chain.npz'): continue
        jobs.append(name)

    if args.num_workers > 1:
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        __map__ = map
    for _ in tqdm.tqdm(__map__(do_job, jobs), total=len(jobs)):
        pass
    if args.num_workers > 1:
        p.__exit__(None, None, None)

# def prot_to_frames(ca_coords, c_coords, n_coords):
#     prot_frames = Rigid.from_3_points(
#         torch.from_numpy(c_coords),
#         torch.from_numpy(ca_coords),
#         torch.from_numpy(n_coords),
#     )
#     rots = torch.eye(3)
#     rots[0, 0] = -1
#     rots[2, 2] = -1
#     rots = Rotation(rot_mats=rots)
#     return prot_frames.compose(Rigid(rots, None))


def traj_to_atom14(traj):
    arr = np.zeros((traj.n_frames, traj.n_residues, 14, 3), dtype=np.dtype(args.dtype))  # npy-dtype-option
    for i, resi in enumerate(traj.top.residues):
        for at in resi.atoms:
            if at.name not in rc.restype_name_to_atom14_names[resi.name]:
                print(resi.name, at.name, 'not found'); continue
            j = rc.restype_name_to_atom14_names[resi.name].index(at.name)
            arr[:,i,j] = traj.xyz[:,at.index] * 10.0
    return arr

if args.atlas:
    def do_job(name):
        for i in args.replicates:
            trajectory_name = (f'{name}_prod_R{i}_fit.xtc' if args.atlas_layout == 'protein'
                               else f'{name}_R{i}.xtc')
            traj = mdtraj.load(f'{args.atlas_dir}/{name}/{trajectory_name}',
                              top=f'{args.atlas_dir}/{name}/{name}.pdb')
            if args.expected_interval_ps is not None:
                if traj.n_frames < 2 or not np.allclose(
                        np.diff(traj.time), args.expected_interval_ps, rtol=0, atol=0.01):
                    raise ValueError(f'{name} R{i}: native frame interval differs from '
                                     f'{args.expected_interval_ps} ps')
            if args.atlas_layout == 'analysis':
                if ''.join(traj.top.to_fasta()) != df.seqres[name]:
                    raise ValueError(f'{name}: topology sequence differs from split seqres')
                if not np.isfinite(traj.xyz).all():
                    raise ValueError(f'{name} R{i}: nonfinite coordinates')
            traj.atom_slice([a.index for a in traj.top.atoms if a.element.symbol != 'H'], True)
            traj.superpose(traj)
            chain_id, residue_index = chain_numbering(traj.top)  # reject before writing anything
            arr = traj_to_atom14(traj)
            stem = f'{args.outdir}/{name}_R{i}{args.suffix}'
            np.save(f'{stem}.npy', arr[::args.stride])
            np.savez(f'{stem}_chain.npz', chain_id=chain_id, residue_index=residue_index)
else:
    def do_job(name):
        traj = mdtraj.load(f'{args.atlas_dir}/{name}/{name}.xtc', top=f'{args.atlas_dir}/{name}/{name}.pdb')
        traj.superpose(traj)
        chain_id, residue_index = chain_numbering(traj.top)  # reject before writing anything
        arr = traj_to_atom14(traj)
        stem = f'{args.outdir}/{name}{args.suffix}'
        np.save(f'{stem}.npy', arr[::args.stride])
        np.savez(f'{stem}_chain.npz', chain_id=chain_id, residue_index=residue_index)

if __name__ == "__main__":
    main()
