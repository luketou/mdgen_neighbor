"""Public attention/model parity against the frozen pre-optimization local model."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent / 'reference'))
from local_attention_before_core_only import run_pretrained_local_attention as reference_attention
from localbaseline.model.latent_model import LatentMDGenModel as ReferenceModel
from mdgen.full_length.attention import run_pretrained_local_attention
from mdgen.full_length.partition import LocalGroup, SpatialPartition
from mdgen.model.latent_model import AttentionWithRoPE
from mdgen.full_length.model import LocalLatentMDGenModel as LatentMDGenModel
from mdgen.rigid_utils import Rigid


def partition(device='cpu'):
    cores = [[0, 3], [1], [2, 4, 5], [6, 8], [7, 9], [10]]
    ghosts = [[8, 2, 1], [5, 3, 0, 7], [9, 0], [2, 10, 1], [6, 5, 3], [8, 4, 0]]
    groups = tuple(LocalGroup((i, 0, 0), torch.tensor(c, device=device), torch.tensor(g, device=device))
                   for i, (c, g) in enumerate(zip(cores, ghosts)))
    owner = torch.full((12,), -1, device=device, dtype=torch.long)
    for i, group in enumerate(groups):
        owner[group.core_indices] = i
    return SpatialPartition(groups, owner, owner >= 0, 3, 4, 16.0)


def assert_parity(reference, candidate, value, kwargs, call):
    a = value.detach().clone().requires_grad_(True)
    b = value.detach().clone().requires_grad_(True)
    expected, actual = call(reference, a, True), call(candidate, b, False)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
    # Nonuniform cotangent catches permutations and avoids a symmetric loss.
    weights = torch.linspace(-1, 1, actual.numel(), device=actual.device).reshape_as(actual)
    (expected * weights).sum().backward()
    (actual * weights).sum().backward()
    torch.testing.assert_close(b.grad, a.grad, atol=3e-5, rtol=1e-4)
    for name, parameter in reference.named_parameters():
        grad = dict(candidate.named_parameters())[name].grad
        if parameter.grad is None:
            assert grad is None, name
        else:
            torch.testing.assert_close(grad, parameter.grad, atol=5e-5, rtol=2e-4, msg=name)


@pytest.mark.parametrize('rope,bias', [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize('group_batch_size', [1, 2, 32])
def test_owned_outputs_and_all_gradients_preserve_old_local_neighborhoods(rope, bias, group_batch_size):
    torch.manual_seed(103)
    model = AttentionWithRoPE(8, 2, add_bias_kv=bias, use_rotary_embeddings=rope, dropout=0).eval()
    p = partition()
    mask = torch.ones(3, 12)
    mask[:, 11] = 0
    mask[0, 1] = 0
    mask[1, 8] = 0
    kwargs = dict(mask=mask, group_batch_size=group_batch_size)
    def call(module, value, old):
        fn = reference_attention if old else run_pretrained_local_attention
        return fn(module, value, p, **kwargs)
    assert_parity(model, copy.deepcopy(model), torch.randn(3, 12, 8), kwargs, call)


def test_model_inference_preserves_local_outputs_and_gradients():
    torch.manual_seed(107)
    config = SimpleNamespace(design=False, tps_condition=False, inpainting=False,
        dynamic_mpnn=False, mpnn=False, embed_dim=8, ipa_head_dim=4, ipa_heads=2,
        ipa_qk=2, ipa_v=2, dropout=0., prepend_ipa=True, interleave_ipa=False,
        no_aa_emb=False, num_layers=2, mha_heads=2, no_rope=False, hyena=False,
        num_frames=3, abs_pos_emb=False, abs_time_emb=False, grad_checkpointing=False,
        time_multiplier=100., sim_condition=True, local_spatial=True,
        cell_summaries=False, group_batch_size=2)
    old, new = ReferenceModel(config, 21).eval(), LatentMDGenModel(config, 21).eval()
    # Default zero output heads/gates would conceal a broken attention path.
    with torch.no_grad():
        for name, param in old.named_parameters():
            if 'cell_summaries' not in name:
                param.normal_(0, .15)
    new.load_state_dict(old.state_dict(), strict=True)
    frame_tensor = torch.zeros(1, 12, 7)
    frame_tensor[..., 0] = 1
    frame_tensor[..., 4:] = torch.randn(1, 12, 3)
    kwargs = dict(mask=torch.ones(1, 3, 12), start_frames=Rigid.from_tensor_7(frame_tensor),
        aatype=torch.zeros(1, 12, dtype=torch.long), spatial_partitions=[partition()],
        x_cond=torch.randn(1, 3, 12, 21), x_cond_mask=torch.zeros(1, 3, 12, dtype=torch.long))
    kwargs['mask'][..., 11] = 0
    def call(model, value, old):
        return model.forward_inference(value, torch.tensor([.5]), **kwargs)
    assert_parity(old, new, torch.randn(1, 3, 12, 21), kwargs, call)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU allocation budget needs CUDA')
@pytest.mark.parametrize('fraction', [.9, .3], ids=['core-output', 'streamed-workspace'])
def test_owned_output_attention_uses_less_peak_memory_than_ghost_output_reference(fraction):
    torch.manual_seed(109)
    device = 'cuda'
    length = 128
    groups = tuple(LocalGroup((i, 0, 0), torch.arange(i*4, i*4+4, device=device),
        torch.tensor([j for j in range(length) if j // 4 != i][:60], device=device)) for i in range(32))
    p = SpatialPartition(groups, torch.arange(length, device=device)//4,
                         torch.ones(length, dtype=torch.bool, device=device), 4, 60, 16.)
    module = AttentionWithRoPE(64, 4, add_bias_kv=True, use_rotary_embeddings=True).cuda().eval()
    value = torch.randn(32, length, 64, device=device)
    def peak(fn):
        with torch.no_grad():
            fn(module, value, p, group_batch_size=2)
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            output = fn(module, value, p, group_batch_size=2)
            torch.cuda.synchronize()
            extra = torch.cuda.max_memory_allocated() - base
            del output
            return extra
    old = peak(reference_attention)
    new = peak(run_pretrained_local_attention)
    print(f'PEAK old={old} new={new} ratio={new/old:.4f}', flush=True)
    assert new < old * fraction, (old, new)
