"""Official checkpoint parity against both frozen model implementations."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch
from scale_sdpa import fixture, move, RESOURCE_ROOT

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/reference'))
from global2026.wrapper import NewMDGenWrapper as FrozenGlobal
from localbaseline.wrapper import NewMDGenWrapper as FrozenLocal
from mdgen.wrapper import NewMDGenWrapper
from mdgen.full_length.checkpoint import load_compatible_state_dict


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--outdir',type=Path,required=True)
    args=parser.parse_args()
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ckpt=RESOURCE_ROOT/'ckpt/atlas.ckpt'
    checkpoint=torch.load(ckpt,map_location='cpu',weights_only=False)
    cases=[]
    for local, reference_class in [(False,FrozenGlobal),(True,FrozenLocal)]:
        config=copy.deepcopy(checkpoint['hyper_parameters']['args'])
        config.local_spatial=local;config.temporal_sdpa=True;config.cell_summaries=False
        config.group_batch_size=4;config.num_frames=250
        reference=reference_class(copy.deepcopy(config)).float().cuda().eval()
        candidate=NewMDGenWrapper(copy.deepcopy(config)).float().cuda().eval()
        for wrapper in (reference,candidate):
            report=load_compatible_state_dict(wrapper,checkpoint)
            assert not report['unexpected'] and not report['shape_mismatch']
            assert all('.cell_summaries.' in name for name in report['newly_initialized'])
        batch,metadata=fixture(104)
        batch=move(batch)
        # Candidate prepares common tensors; frozen models consume the same bytes.
        prepared=candidate.prep_batch(batch)
        kwargs=prepared['model_kwargs']
        value=prepared['latents'].clone()
        short={key: (v[:,:4] if key in ('mask','x_cond','x_cond_mask') else v)
               for key,v in kwargs.items()}
        torch.manual_seed(137)
        x0=torch.randn_like(value[:,:4]); t=torch.tensor([.5],device='cuda')
        outputs=[];gradients=[]
        for model in (reference.model,candidate.model):
            x=x0.detach().clone().requires_grad_(True)
            out=model.forward_inference(x,t,**short)
            cot=torch.linspace(-1,1,out.numel(),device='cuda').reshape_as(out)/out.numel()
            params=dict(model.named_parameters())
            grad=torch.autograd.grad((out*cot).sum(),[x,*params.values()],allow_unused=True)
            outputs.append(out.detach());gradients.append(dict(zip(['input',*params],grad)))
        torch.testing.assert_close(outputs[1],outputs[0],atol=2e-5,rtol=2e-4)
        max_weight=0.
        for name,old in gradients[0].items():
            new=gradients[1][name]
            if old is None:assert new is None,name
            else:
                torch.testing.assert_close(new,old,atol=2e-5,rtol=2e-4,msg=name)
                if name!='input':max_weight=max(max_weight,float((new-old).abs().max()))
        short_output_error=float((outputs[1]-outputs[0]).abs().max())
        input_error=float((gradients[1]['input']-gradients[0]['input']).abs().max())
        del outputs,gradients
        with torch.no_grad():
            torch.manual_seed(137);x=torch.randn_like(value)
            old=reference.model.forward_inference(x,t,**kwargs)
            new=candidate.model.forward_inference(x,t,**kwargs)
            torch.testing.assert_close(new,old,atol=2e-5,rtol=2e-4)
            assert torch.isfinite(new).all()
            cases.append(dict(mode='local' if local else 'global',status='passed',
                gradient_frames=4,inference_frames=250,residues=104,
                forward_max_abs_difference=float((new-old).abs().max()),
                gradient_forward_max_abs_difference=short_output_error,
                input_gradient_max_abs_difference=input_error,
                parameter_gradient_max_abs_difference=max_weight,fixture=metadata))
        del reference,candidate,value,kwargs,batch,old,new
        torch.cuda.empty_cache()
    result=dict(status='passed',checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),cases=cases)
    args.outdir.mkdir(parents=True,exist_ok=True)
    (args.outdir/'checkpoint_parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
