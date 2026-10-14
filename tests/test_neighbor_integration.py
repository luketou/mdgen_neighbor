"""Behavioral parity at the agreed model forward_inference seam."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests/reference'))
from global2026.model.latent_model import LatentMDGenModel as FrozenGlobal
from localbaseline.model.latent_model import LatentMDGenModel as FrozenLocal
from mdgen.model.latent_model import LatentMDGenModel
from mdgen.wrapper import NewMDGenWrapper
from mdgen.rigid_utils import Rigid


def config(local=False, rope=True):
    return SimpleNamespace(design=False,tps_condition=False,inpainting=False,
        dynamic_mpnn=False,mpnn=False,embed_dim=32,ipa_head_dim=4,ipa_heads=2,
        ipa_qk=2,ipa_v=2,dropout=0.,prepend_ipa=True,interleave_ipa=True,
        no_aa_emb=False,num_layers=1,mha_heads=4,no_rope=not rope,hyena=False,
        num_frames=4,abs_pos_emb=False,abs_time_emb=False,grad_checkpointing=False,
        time_multiplier=100.,sim_condition=True,local_spatial=local,
        temporal_sdpa=True,cell_summaries=False,group_batch_size=2,
        spatial_cell_width=4.,spatial_contact_radius=8.,core_capacity=4,
        ghost_capacity=20,ffn_chunk_size=65536)


def check_parity(reference, candidate, kwargs, training):
    torch.manual_seed(913)
    with torch.no_grad():
        for name,p in reference.named_parameters():
            if 'cell_summaries' not in name:p.normal_(0,.08)
    candidate.load_state_dict(reference.state_dict(),strict=True)
    reference.train(training);candidate.train(training)
    value=torch.randn(1,4,12,21);t=torch.tensor([.43])
    outputs=[];grads=[]
    for model in (reference,candidate):
        x=value.clone().requires_grad_(True)
        out=model.forward_inference(x,t,**kwargs)
        cot=torch.linspace(-1,1,out.numel()).reshape_as(out)/out.numel()
        params=dict(model.named_parameters())
        gradient=torch.autograd.grad((out*cot).sum(),[x,*params.values()],allow_unused=True)
        outputs.append(out.detach());grads.append(dict(zip(['input',*params],gradient)))
    torch.testing.assert_close(outputs[0],outputs[1],atol=2e-5,rtol=2e-4)
    for name,old in grads[0].items():
        new=grads[1][name]
        if old is None:assert new is None,name
        else:torch.testing.assert_close(old,new,atol=2e-5,rtol=2e-4,msg=name)
    with torch.no_grad():
        torch.testing.assert_close(reference.forward_inference(value,t,**kwargs),
            candidate.forward_inference(value,t,**kwargs),atol=2e-5,rtol=2e-4)


def inputs():
    frames=Rigid.identity((1,12),device=torch.device('cpu'))
    trans=torch.arange(12,dtype=torch.float32)[None,:,None].expand(1,12,3)*2
    frames=Rigid(frames.get_rots(),trans)
    return dict(mask=torch.ones(1,4,12),start_frames=frames,
        aatype=torch.arange(12)[None])


@pytest.mark.parametrize('rope',[False,True])
@pytest.mark.parametrize('training',[False,True])
def test_global_preserves_frozen_2026_output_and_all_gradients(rope,training):
    args=config(rope=rope)
    check_parity(FrozenGlobal(copy.deepcopy(args),21),LatentMDGenModel(args,21),inputs(),training)


@pytest.mark.parametrize('rope',[False,True])
@pytest.mark.parametrize('training',[False,True])
def test_local_wrapper_preserves_frozen_mdgen_output_and_all_gradients(rope,training):
    args=config(local=True,rope=rope)
    args.path_type='GVP';args.prediction='velocity';args.ema=False;args.no_frames=False
    args.design_key_frames=False;args.no_offsets=False;args.no_torsion=False
    args.dirichlet_flow_temp=1.;args.sampling_method='dopri5'
    candidate=NewMDGenWrapper(args).model
    # Default global is already tested separately; this must exercise local groups.
    check_parity(FrozenLocal(copy.deepcopy(args),21),candidate,inputs()|dict(
        chain_id=torch.zeros(1,12,dtype=torch.long),
        residue_index=torch.arange(12)[None]),training)


@pytest.mark.parametrize('local',[False,True])
@pytest.mark.parametrize('training',[False,True])
def test_inference_ffn_chunking_preserves_full_outputs_and_gradient_fallback(local,training):
    from mdgen.full_length.model import LocalLatentMDGenModel
    args=config(local=local)
    reference_type=FrozenLocal if local else FrozenGlobal
    candidate_type=LocalLatentMDGenModel if local else LatentMDGenModel
    reference=reference_type(copy.deepcopy(args),21)
    candidate=candidate_type(copy.deepcopy(args),21)
    # 48 tokens: the candidate must split into multiple inference chunks,
    # while the frozen reference's 65536 threshold computes the full FFN.
    for layer in candidate.layers:layer.ffn_chunk_size=7
    kwargs=inputs()
    if local:
        kwargs.update(chain_id=torch.zeros(1,12,dtype=torch.long),
                      residue_index=torch.arange(12)[None])
    check_parity(reference,candidate,kwargs,training)


@pytest.mark.parametrize('training',[False,True])
def test_local_grad_checkpointing_is_used_and_preserves_outputs_and_gradients(training,monkeypatch):
    from mdgen.full_length.model import LocalLatentMDGenModel
    import mdgen.model.latent_model as latent_model
    calls=[]
    original=latent_model.checkpoint
    def counting_checkpoint(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(latent_model,'checkpoint',counting_checkpoint)
    reference=LocalLatentMDGenModel(config(local=True),21)
    args=config(local=True);args.grad_checkpointing=True
    candidate=LocalLatentMDGenModel(args,21)
    kwargs=inputs()|dict(chain_id=torch.zeros(1,12,dtype=torch.long),
        residue_index=torch.arange(12)[None])
    check_parity(reference,candidate,kwargs,training)
    assert calls


def test_fill_local_defaults_matches_cli_defaults_and_keeps_saved_values(monkeypatch):
    from mdgen.parsing import LOCAL_DEFAULTS, fill_local_defaults, parse_train_args
    monkeypatch.setattr(sys,'argv',['train.py','--train_split','a.csv',
        '--val_split','b.csv','--data_dir','data'])
    monkeypatch.delenv('MODEL_DIR',raising=False)
    cli=parse_train_args()
    for key in ('local_spatial','cell_summaries','spatial_cell_width',
                'spatial_contact_radius','core_capacity','ghost_capacity','group_batch_size'):
        assert getattr(cli,key)==LOCAL_DEFAULTS[key],key
    # Checkpoints saved before the local options keep their explicit values.
    saved=SimpleNamespace(core_capacity=7)
    fill_local_defaults(saved)
    assert saved.core_capacity==7
    assert saved.ghost_capacity==LOCAL_DEFAULTS['ghost_capacity']
    assert saved.local_spatial is False


def test_local_model_accepts_args_saved_without_local_options():
    from mdgen.full_length.model import LocalLatentMDGenModel
    from mdgen.parsing import LOCAL_DEFAULTS
    args=config(local=True)
    for key in LOCAL_DEFAULTS:
        if key!='local_spatial' and hasattr(args,key):delattr(args,key)
    model=LocalLatentMDGenModel(args,21).eval()
    assert model.layers[0].group_batch_size==LOCAL_DEFAULTS['group_batch_size']
    with torch.no_grad():
        out=model.forward_inference(torch.randn(1,4,12,21),torch.tensor([.43]),**inputs())
    assert torch.isfinite(out).all()


def test_partitions_default_to_single_chain_position_numbering():
    from mdgen.full_length.partition import partitions_from_args
    from mdgen.parsing import fill_local_defaults
    args=fill_local_defaults(config(local=True))
    frames=inputs()['start_frames']
    mask=torch.ones(1,12)
    implicit=partitions_from_args(args,frames.get_trans(),mask)
    explicit=partitions_from_args(args,frames.get_trans(),mask,
        chain_id=torch.zeros(1,12,dtype=torch.long),residue_index=torch.arange(12)[None])
    assert len(implicit)==len(explicit)==1
    assert len(implicit[0].groups)==len(explicit[0].groups)>1
    for a,b in zip(implicit[0].groups,explicit[0].groups):
        assert torch.equal(a.core_indices,b.core_indices)
        assert torch.equal(a.ghost_indices,b.ghost_indices)
