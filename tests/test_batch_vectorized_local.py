"""Batch-vectorized local layers must match the former per-sample loops."""
import torch
import pytest

import mdgen.full_length.model as local_model
from mdgen.full_length.attention import run_pretrained_local_attention
from mdgen.full_length.local_ipa import run_local_ipa
from mdgen.full_length.model import LocalIPALayer, LocalLayer
from mdgen.full_length.partition import LocalGroup, SpatialPartition, merge_partitions
from mdgen.rigid_utils import Rigid

L, C, B, T = 12, 16, 3, 4
IPA = dict(c_s=C, c_z=0, c_hidden=4, no_heads=2, no_qk_points=2, no_v_points=2, dropout=0.0)


def make_partition(cores, ghosts):
    groups = tuple(LocalGroup((i, 0, 0), torch.tensor(c), torch.tensor(g))
                   for i, (c, g) in enumerate(zip(cores, ghosts)))
    owner = torch.full((L,), -1, dtype=torch.long)
    for i, g in enumerate(groups):
        owner[g.core_indices] = i
    return SpatialPartition(groups, owner, owner >= 0, 3, 4, 16.0)


def partitions():
    p0 = make_partition([[0, 3], [1], [2, 4, 5], [6, 8], [7, 9], [10]],
                        [[8, 2, 1], [5, 3, 0, 7], [9, 0], [2, 10, 1], [6, 5, 3], [8, 4, 0]])
    p1 = make_partition([[5, 1, 0], [2, 3, 4], [6, 7, 8, 9]],
                        [[11, 2], [0, 7, 10], [1]])
    p2 = make_partition([[i] for i in range(11)], [[(i + 1) % 11] for i in range(11)])
    return [p0, p1, p2]


def frames(n):
    t = torch.randn(n, L, 7)
    t[..., :4] = torch.nn.functional.normalize(t[..., :4], dim=-1)
    return Rigid.from_tensor_7(t)


def rand_mask(*shape):
    m = torch.ones(*shape)
    m[..., 11] = 0
    m[..., 2] = (torch.rand(*shape[:-1]) > 0.3).float()
    return m


def test_merge_partitions_offsets():
    ps = partitions()
    merged = merge_partitions(ps)
    assert len(merged.groups) == sum(len(p.groups) for p in ps)
    assert merged.residue_mask.numel() == B * L
    assert merged.validate_ownership().valid
    assert merged.diagnostics()["owner_group_matches"]


@pytest.mark.parametrize('gbs', [1, 2, 32])
def test_local_layer_matches_loop_and_batches_kernels(gbs, monkeypatch):
    torch.manual_seed(0)
    layer = LocalLayer(C, 4 * C, 2, dropout=0, num_frames=T, use_rotary_embeddings=True,
                       ipa_args=IPA, group_batch_size=gbs).eval()
    ps = partitions()
    x = torch.randn(B, T, L, C)
    mask = rand_mask(B, T, L)
    fr = frames(B)
    with torch.no_grad():
        att_ref = torch.stack([run_pretrained_local_attention(
            layer.mha_l, x[b], ps[b], mask=mask[b], group_batch_size=gbs)
            for b in range(B)])
        ipa_ref = []
        for b in range(B):
            f = Rigid.from_tensor_7(fr[b:b + 1].to_tensor_7()[:, None].expand(-1, T, -1, -1).reshape(T, L, 7))
            ipa_ref.append(run_local_ipa(layer.ipa, layer.ipa_norm(x[b]), f, ps[b], group_batch_size=gbs))
        ipa_ref = torch.stack(ipa_ref)
    calls = {'att': 0, 'ipa': 0}
    for name, key, fn in (('run_pretrained_local_attention', 'att', run_pretrained_local_attention),
                          ('run_local_ipa', 'ipa', run_local_ipa)):
        def counted(*a, _k=key, _f=fn, **kw):
            calls[_k] += 1
            return _f(*a, **kw)
        monkeypatch.setattr(local_model, name, counted)
    with torch.no_grad():
        att = layer._spatial_attention(x, mask, ps)
        ipa = layer._spatial_ipa(x, fr, mask, ps)
    assert calls == {'att': 1, 'ipa': 1}
    torch.testing.assert_close(att, att_ref, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(ipa, ipa_ref, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize('gbs', [1, 2, 32])
def test_local_ipa_layer_matches_loop(gbs):
    torch.manual_seed(1)
    layer = LocalIPALayer(C, 4 * C, 2, dropout=0, use_rotary_embeddings=True,
                          ipa_args=IPA, group_batch_size=gbs).eval()
    ps = partitions()
    x = torch.randn(B, L, C)
    mask = rand_mask(B, L)
    fr = frames(B)
    with torch.no_grad():
        att_ref = torch.cat([run_pretrained_local_attention(
            layer.mha_l, x[b:b + 1], ps[b], mask=mask[b:b + 1], group_batch_size=gbs)
            for b in range(B)])
        ipa_ref = torch.cat([run_local_ipa(
            layer.ipa, layer.ipa_norm(x[b:b + 1]), fr[b:b + 1], ps[b], group_batch_size=gbs)
            for b in range(B)])
        att = layer._spatial_attention(x, mask, ps)
        ipa = layer._spatial_ipa(x, fr, mask, ps)
    torch.testing.assert_close(att, att_ref, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(ipa, ipa_ref, atol=2e-5, rtol=1e-4)


def test_local_layer_gradients_match_loop():
    torch.manual_seed(2)
    layer = LocalLayer(C, 4 * C, 2, dropout=0, num_frames=T, use_rotary_embeddings=True,
                       ipa_args=IPA, group_batch_size=2).eval()
    ps = partitions()
    mask = rand_mask(B, T, L)
    fr = frames(B)
    x0 = torch.randn(B, T, L, C)
    w = torch.randn(B, T, L, C)

    def loop(x):
        att = torch.stack([run_pretrained_local_attention(
            layer.mha_l, x[b], ps[b], mask=mask[b], group_batch_size=2) for b in range(B)])
        ipa = []
        for b in range(B):
            f = Rigid.from_tensor_7(fr[b:b + 1].to_tensor_7()[:, None].expand(-1, T, -1, -1).reshape(T, L, 7))
            ipa.append(run_local_ipa(layer.ipa, layer.ipa_norm(x[b]), f, ps[b], group_batch_size=2))
        return att, torch.stack(ipa)

    def grads(fn):
        layer.zero_grad()
        x = x0.clone().requires_grad_(True)
        att, ipa = fn(x)
        ((att + ipa) * w).sum().backward()
        return x.grad, {n: p.grad.clone() for n, p in layer.named_parameters() if p.grad is not None}

    gx_ref, gp_ref = grads(loop)
    gx, gp = grads(lambda x: (layer._spatial_attention(x, mask, ps), layer._spatial_ipa(x, fr, mask, ps)))
    torch.testing.assert_close(gx, gx_ref, atol=1e-4, rtol=1e-3)
    assert gp.keys() == gp_ref.keys() and gp
    for n in gp:
        torch.testing.assert_close(gp[n], gp_ref[n], atol=1e-4, rtol=1e-3)
