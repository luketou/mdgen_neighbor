"""Official ATLAS checkpoint: fifty saved points versus endpoint inference."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from mdgen.wrapper import NewMDGenWrapper
from mdgen.full_length.checkpoint import load_compatible_state_dict
from scale_sdpa import fixture, move

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/reference'))
from wrapper_before_ode_endpoints import FiftyPointWrapper


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    args=parser.parse_args()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    checkpoint=Path(__import__('os').environ.get('MDGEN_RESOURCE_ROOT', str(ROOT.parent/'mdgen')))/'ckpt/atlas.ckpt'
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=copy.deepcopy(ckpt['hyper_parameters']['args'])
    config.local_spatial,config.temporal_sdpa,config.cell_summaries=True,True,False
    config.group_batch_size,config.ffn_chunk_size,config.num_frames=4,65536,250
    config.sampling_method='dopri5'
    torch.manual_seed(137)
    wrapper=NewMDGenWrapper(config).float().cuda().eval()
    loading=load_compatible_state_dict(wrapper,ckpt)
    assert not loading['shape_mismatch'] and not loading['unexpected']
    assert all('.cell_summaries.' in k for k in loading['newly_initialized'])
    del ckpt
    batch, metadata=fixture(104)
    batch=move(batch)
    counts=[0];saved=[]
    forward=wrapper.model.forward_inference
    def counted(*args,**kwargs):
        counts[0]+=1
        return forward(*args,**kwargs)
    wrapper.model.forward_inference=counted
    sampler=wrapper.transport_sampler.sample_ode
    def observed_sampler(*args,**kwargs):
        sample=sampler(*args,**kwargs)
        def observed(*args,**kwargs):
            result=sample(*args,**kwargs)
            saved.append(len(result))
            return result
        return observed
    wrapper.transport_sampler.sample_ode=observed_sampler
    outputs,nfe=[],[]
    with torch.no_grad():
        for inference in (FiftyPointWrapper.inference,NewMDGenWrapper.inference):
            counts[0]=0
            torch.manual_seed(137);torch.cuda.manual_seed_all(137)
            output=inference(wrapper,batch)
            outputs.append(tuple(v.cpu() for v in output));nfe.append(counts[0])
            del output
    torch.testing.assert_close(outputs[1][0],outputs[0][0],atol=2e-5,rtol=1e-4)
    torch.testing.assert_close(outputs[1][1],outputs[0][1],atol=0,rtol=0)
    assert nfe[0]==nfe[1] and saved==[50,2]
    finite=bool(outputs[1][0].isfinite().all());assert finite
    report=dict(status='passed',checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),frames=250,residues=104,precision='float32',tf32=False,
        seed=137,sampling_method='dopri5',atol=1e-6,rtol=1e-3,ffn_chunk_size=65536,
        saved_points=saved,model_evaluations=nfe,output_shape=list(outputs[1][0].shape),
        output_max_abs_difference=float((outputs[1][0]-outputs[0][0]).abs().max()),
        output_exact_equal=bool(torch.equal(outputs[1][0],outputs[0][0])),output_all_finite=finite,fixture=metadata)
    args.outdir.mkdir(parents=True,exist_ok=True)
    (args.outdir/'checkpoint_parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
