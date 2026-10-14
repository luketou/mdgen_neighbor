import argparse
import copy
parser = argparse.ArgumentParser()
parser.add_argument('--sim_ckpt', type=str, default=None, required=True)
parser.add_argument('--data_dir', type=str, default=None, required=True)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--num_frames', type=int, default=1000)
parser.add_argument('--num_rollouts', type=int, default=100)
parser.add_argument('--no_frames', action='store_true')
parser.add_argument('--tps', action='store_true')
parser.add_argument('--xtc', action='store_true')
parser.add_argument('--save_atom14', action='store_true', help='Save reconstructed coordinates before PDB/XTC export for diagnostics.')
parser.add_argument('--out_dir', type=str, default=".")
parser.add_argument('--split', type=str, default='splits/4AA_test.csv')
parser.add_argument('--inference_seed', type=int, default=None)  # seed-deterministic-args
parser.add_argument('--ode_atol', type=float, default=1e-6)
parser.add_argument('--ode_rtol', type=float, default=1e-3)
parser.add_argument('--manual_attention', action='store_true')  # sdpa-route
parser.add_argument('--local_spatial', action=argparse.BooleanOptionalAction, default=None)
parser.add_argument('--cell_summaries', action=argparse.BooleanOptionalAction, default=None)
parser.add_argument('--spatial_cell_width', type=float, default=None)
parser.add_argument('--spatial_contact_radius', type=float, default=None)
parser.add_argument('--core_capacity', type=int, default=None)
parser.add_argument('--ghost_capacity', type=int, default=None)
parser.add_argument('--group_batch_size', type=int, default=None)

# seed-deterministic-args:start
parser.add_argument(
    '--deterministic',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Use deterministic algorithms when available.',
)
parser.add_argument(
    '--benchmark',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Enable the cuDNN benchmark autotuner.',
)
# seed-deterministic-args:end
args = parser.parse_args()
if not (0 < args.ode_atol < float('inf') and 0 < args.ode_rtol < float('inf')):
    parser.error('ODE tolerances must be positive and finite')
# deterministic-benchmark-guard:start
if args.deterministic and args.benchmark:
    parser.error('--deterministic and --benchmark cannot both be enabled')
# deterministic-benchmark-guard:end

import os, torch, mdtraj, tqdm, time
import numpy as np
from pytorch_lightning import seed_everything  # seed-initialization
from mdgen.geometry import atom14_to_frames, atom14_to_atom37, atom37_to_torsions
from mdgen.residue_constants import restype_order, restype_atom37_mask
from mdgen.tensor_utils import tensor_tree_map
from mdgen.wrapper import NewMDGenWrapper
from mdgen.utils import atom14_to_pdb
import pandas as pd

# seed-initialization:start
if args.inference_seed is not None:
    seed_everything(args.inference_seed, workers=True)
# seed-initialization:end
torch.use_deterministic_algorithms(args.deterministic)  # deterministic-execution
torch.backends.cudnn.benchmark = args.benchmark  # deterministic-benchmark-guard


os.makedirs(args.out_dir, exist_ok=True)


def get_batch(name, seqres, num_frames):
    arr = np.lib.format.open_memmap(f'{args.data_dir}/{name}{args.suffix}.npy', 'r')

    if not args.tps: # else keep all frames
        arr = np.copy(arr[0:1]).astype(np.float32)

    if arr.shape[1] != len(seqres):
        raise ValueError(
            f'{name}: NPY residue count {arr.shape[1]} does not match '
            f'CSV seqres length {len(seqres)}'
        )

    frames = atom14_to_frames(torch.from_numpy(arr))
    seqres = torch.tensor([restype_order[c] for c in seqres])
    atom37 = torch.from_numpy(atom14_to_atom37(arr, seqres[None])).float()
    L = len(seqres)
    mask = torch.ones(L)
    
    if args.no_frames:
        return {
            'atom37': atom37,
            'seqres': seqres,
            'mask': restype_atom37_mask[seqres],
        }
        
    torsions, torsion_mask = atom37_to_torsions(atom37, seqres[None])
    return {
        'torsions': torsions,
        'torsion_mask': torsion_mask[0],
        'trans': frames._trans,
        'rots': frames._rots._rot_mats,
        'seqres': seqres,
        'mask': mask, # (L,)
    }

def rollout(model, batch):

    #print('Start sim', batch['trans'][0,0,0])
    if args.no_frames:
        
        expanded_batch = {
            'atom37': batch['atom37'].expand(-1, args.num_frames, -1, -1, -1),
            'seqres': batch['seqres'],
            'mask': batch['mask'],
        }
    else:    
        expanded_batch = {
            'torsions': batch['torsions'].expand(-1, args.num_frames, -1, -1, -1),
            'torsion_mask': batch['torsion_mask'],
            'trans': batch['trans'].expand(-1, args.num_frames, -1, -1),
            'rots': batch['rots'].expand(-1, args.num_frames, -1, -1, -1),
            'seqres': batch['seqres'],
            'mask': batch['mask'],
        }
    atom14, _ = model.inference(expanded_batch)
    new_batch = {**batch}

    if args.no_frames:
        new_batch['atom37'] = torch.from_numpy(
            atom14_to_atom37(atom14[:,-1].cpu(), batch['seqres'][0].cpu())
        ).cuda()[:,None].float()
        
        
        
    else:
        frames = atom14_to_frames(atom14[:,-1])
        new_batch['trans'] = frames._trans[None]
        new_batch['rots'] = frames._rots._rot_mats[None]
        atom37 = atom14_to_atom37(atom14[0,-1].cpu(), batch['seqres'][0].cpu())
        torsions, _ = atom37_to_torsions(atom37, batch['seqres'][0].cpu())
        new_batch['torsions'] = torsions[None, None].cuda()

    return atom14, new_batch
    
    
def do(model, name, seqres):

    item = get_batch(name, seqres, num_frames=model.args.num_frames)
    batch = next(iter(torch.utils.data.DataLoader([item])))

    batch = tensor_tree_map(lambda x: x.cuda(), batch)  
    
    all_atom14 = []
    start = time.time()
    for _ in tqdm.trange(args.num_rollouts):
        atom14, batch = rollout(model, batch)
        # print(atom14[0,0,0,1], atom14[0,-1,0,1])
        all_atom14.append(atom14)

    print(time.time() - start)
    all_atom14 = torch.cat(all_atom14, 1)
    
    if args.save_atom14:
        np.save(os.path.join(args.out_dir, f'{name}_atom14.npy'), all_atom14[0].cpu().numpy())
    path = os.path.join(args.out_dir, f'{name}.pdb')
    atom14_to_pdb(all_atom14[0].cpu().numpy(), batch['seqres'][0].cpu().numpy(), path)

    if args.xtc:
        traj = mdtraj.load(path)
        traj.superpose(traj)
        traj.save(os.path.join(args.out_dir, f'{name}.xtc'))
        traj[0].save(os.path.join(args.out_dir, f'{name}.pdb'))

@torch.no_grad()
def main():
    from mdgen.full_length.checkpoint import load_compatible_state_dict
    checkpoint = torch.load(args.sim_ckpt, map_location='cpu', weights_only=False)
    model_args = copy.deepcopy(checkpoint['hyper_parameters']['args'])
    for key in ('local_spatial', 'cell_summaries', 'spatial_cell_width',
                'spatial_contact_radius', 'core_capacity', 'ghost_capacity', 'group_batch_size'):
        if getattr(args, key) is not None:
            setattr(model_args, key, getattr(args, key))
    if args.no_frames != bool(getattr(model_args, 'no_frames', False)):
        parser.error('--no_frames must match the checkpoint architecture')
    if getattr(model_args, 'local_spatial', False) and (args.manual_attention or args.no_frames):
        parser.error('local_spatial requires SDPA and residue frames')
    if getattr(model_args, 'abs_time_emb', False) and args.num_frames != model_args.num_frames:
        parser.error('fixed absolute time embeddings require the checkpoint num_frames')
    model = NewMDGenWrapper(model_args, use_sdpa=not args.manual_attention)
    report = load_compatible_state_dict(model, checkpoint)
    missing = [name for name in report['newly_initialized'] if '.cell_summaries.' not in name]
    if missing or report['shape_mismatch'] or report['unexpected']:
        raise ValueError(f'Checkpoint incompatible: {report}')
    print(f'Checkpoint load: {report}', flush=True)
    model.args.ode_atol = args.ode_atol
    model.args.ode_rtol = args.ode_rtol
    print(f'ODE: {model.args.sampling_method}, atol={args.ode_atol}, rtol={args.ode_rtol}', flush=True)
    model.eval().float().to('cuda')  # fp32-inference-model
    
    df = pd.read_csv(args.split, index_col='name')
    for name in df.index:
        if args.pdb_id and name not in args.pdb_id:
            continue
        seqres = df.seqres[name]
        do(model, name, seqres)
        

main()
