"""Time and peak memory of local forward_inference for B > 1 (#3 Ticket 1).

Run once per code version on the same card; --repo selects which checkout's
`mdgen` is imported, so before/after use identical inputs and this script.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import torch

parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--repo',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--batch_sizes',type=int,nargs='+',default=[1,2,4,8])
parser.add_argument('--residues',type=int,default=104)
parser.add_argument('--frames',type=int,default=250)
parser.add_argument('--repeats',type=int,default=5)
args=parser.parse_args()
sys.path.insert(0,str(args.repo.resolve()))
sys.path.insert(1,str(Path(__file__).resolve().parent))
from scale_sdpa import fixture, move, RESOURCE_ROOT
from mdgen.wrapper import NewMDGenWrapper
from mdgen.full_length.checkpoint import load_compatible_state_dict

torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
checkpoint=torch.load(RESOURCE_ROOT/'ckpt/atlas.ckpt',map_location='cpu',weights_only=False)
config=copy.deepcopy(checkpoint['hyper_parameters']['args'])
config.local_spatial=True;config.temporal_sdpa=True;config.cell_summaries=False
config.group_batch_size=4;config.num_frames=args.frames
wrapper=NewMDGenWrapper(copy.deepcopy(config)).float().cuda().eval()
load_compatible_state_dict(wrapper,checkpoint)
batch,metadata=fixture(args.residues)
batch=move(batch)
t=torch.tensor([.5],device='cuda')


def prepare(B):
    # Repeat the raw batch so prep_batch builds every per-sample tensor/frame.
    repeated={k:(torch.cat([v]*B) if torch.is_tensor(v) else v) for k,v in batch.items()}
    prepared=wrapper.prep_batch(repeated)
    for k,v in [('latents',prepared['latents']),*prepared['model_kwargs'].items()]:
        if hasattr(v,'shape'):
            assert v.shape[0]==B,f'{k}: batch dim {v.shape[0]} != {B}'
    return prepared['latents'],prepared['model_kwargs']


def select(kwargs,b):
    return {k:(v[b:b+1] if hasattr(v,'__getitem__') and not isinstance(v,(bool,int,float)) else v)
            for k,v in kwargs.items()}


results=[];reference=None
with torch.no_grad():
    for B in args.batch_sizes:
        torch.manual_seed(137)
        latents,kwargs=prepare(B)
        x=torch.randn_like(latents)
        tt=t.expand(B)
        out=wrapper.model.forward_inference(x,tt,**kwargs)  # warm-up
        # Per-sample loop on the same inputs: vectorized B > 1 must match it.
        loop=torch.cat([wrapper.model.forward_inference(
            x[b:b+1],tt[b:b+1],**select(kwargs,b))
            for b in range(B)])
        loop_error=float((out-loop).abs().max())
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        times=[]
        for _ in range(args.repeats):
            torch.cuda.synchronize();start=time.perf_counter()
            out=wrapper.model.forward_inference(x,tt,**kwargs)
            torch.cuda.synchronize();times.append(time.perf_counter()-start)
        times.sort()
        results.append(dict(batch_size=B,median_seconds=times[len(times)//2],
            min_seconds=times[0],peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            loop_max_abs_difference=loop_error,output_all_finite=bool(torch.isfinite(out).all())))
        print(json.dumps(results[-1]),flush=True)
        del out,loop,x,kwargs;torch.cuda.empty_cache()
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(dict(repo=str(args.repo.resolve()),gpu=torch.cuda.get_device_name(0),
    gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),residues=args.residues,frames=args.frames,
    group_batch_size=4,precision='float32',tf32=False,repeats=args.repeats,fixture=metadata,
    results=results),indent=2)+'\n')
