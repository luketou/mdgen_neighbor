"""
【檔案功能說明】
本檔案 (latent_model.py) 是 MDGen 專案的核心 AI 神經網路結構 (`LatentMDGenModel`)。
主要負責：
1. 接收分子軌跡在流匹配過程中的當前狀態 $x_t$ (含剛體變換 7 維向量、扭轉角 14 維向量及殘基序列)。
2. 將時間步 t、殘基位置編碼、殘基種類 Embedding 以及相對剛體位移偏差向量 (offsets) 融合編碼。
3. 通過交錯堆疊的不變點注意力層 (IPA) 與多頭注意力層 (MHA / Hyena) 進行 3D 空間與時間軸特徵融合。
4. 預輸出速度場 (Velocity Field): 包含平移速度 $\vec{v}_x$、旋轉變換 $\vec{v}_R$ 與 7 個二面扭轉角的變化率 $\vec{v}_\theta$。
"""

import torch, tqdm, math
from torch.utils.checkpoint import checkpoint

from .standalone_hyena import HyenaOperator
from ..transport.transport import t_to_alpha
from .mha import MultiheadAttention
import numpy as np
import torch.nn as nn
from .layers import GaussianFourierProjection, TimestepEmbedder, FinalLayer
from .layers import gelu, modulate
from .ipa import InvariantPointAttention
from ..utils import DirichletConditionalFlow, simplex_proj, get_offsets
from ..rigid_utils import Rigid
from ..full_length.attention import run_pretrained_local_attention, run_pretrained_sdpa_attention
from ..full_length.local_ipa import run_local_ipa
from ..full_length.partition import build_spatial_partition
from ..full_length.summaries import CoreCellSummaryExchange


def grad_checkpoint(func, args, checkpointing=False):
    """
    【函式說明】
    梯度檢查點 (Gradient Checkpointing) 封裝函式，用於在訓練時節省 GPU 記憶體。
    """
    if checkpointing:
        return checkpoint(func, *args, use_reentrant=False)
    else:
        return func(*args)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    【函式說明】
    計算一維正弦/餘弦 (Sin/Cos) 位置嵌入矩陣。
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class LatentMDGenModel(nn.Module):
    """
    【類別說明】
    MDGen 潛在軌跡生成模型的核心神經網路。
    結合 IPA 空間幾何注意力與 MHA/Hyena 時空序列注意力，預測分子軌跡流匹配的速度場。
    """
    def __init__(self, args, latent_dim):
        """
        【初始化函式】
        建構模型所有的 Embedding 層、IPA 注意力層、MHA 注意力層與速度場預測頭 (Output Heads)。
        """
        super().__init__()
        self.args = args
        if self.args.design:
            assert self.args.prepend_ipa

        self.latent_to_emb = nn.Linear(latent_dim, args.embed_dim)
        if self.args.tps_condition or self.args.inpainting or self.args.dynamic_mpnn:
            self.latent_to_emb_f = nn.Linear(7, args.embed_dim)
            self.latent_to_emb_r = nn.Linear(7, args.embed_dim)

        cond_dim = latent_dim
        if self.args.design: cond_dim -= 20
        self.cond_to_emb = nn.Linear(cond_dim, args.embed_dim)
        self.mask_to_emb = nn.Embedding(2, args.embed_dim)
        if self.args.design:
            self.x_d_to_emb = nn.Linear(20, args.embed_dim)

        ipa_args = {
            'c_s': args.embed_dim,
            'c_z': 0,
            'c_hidden': args.ipa_head_dim,
            'no_heads': args.ipa_heads,
            'no_qk_points': args.ipa_qk,
            'no_v_points': args.ipa_v,
            'dropout': args.dropout,
        }

        if args.prepend_ipa:
            if not self.args.no_aa_emb:
                self.aatype_to_emb = nn.Embedding(21, args.embed_dim)
            self.ipa_layers = nn.ModuleList(
                [
                    IPALayer(
                        embed_dim=args.embed_dim,
                        ffn_embed_dim=4 * args.embed_dim,
                        mha_heads=args.mha_heads,
                        dropout=args.dropout,
                        use_rotary_embeddings=not args.no_rope,
                        ipa_args=ipa_args,
                        local_spatial=getattr(args, 'local_spatial', False),
                        attention_backend=getattr(args, 'spatial_attention_backend', 'sdpa'),
                        group_batch_size=getattr(args, 'group_batch_size', 4),
                        cell_summaries_enabled=getattr(args, 'cell_summaries', True),
                    )
                    for _ in range(args.num_layers)
                ]
            )

        self.layers = nn.ModuleList(
            [
                LatentMDGenLayer(
                    embed_dim=args.embed_dim,
                    ffn_embed_dim=4 * args.embed_dim,
                    mha_heads=args.mha_heads,
                    dropout=args.dropout,
                    hyena=args.hyena,
                    num_frames=args.num_frames,
                    use_rotary_embeddings=not args.no_rope,
                    use_time_attention=True,
                    temporal_sdpa=getattr(args, 'temporal_sdpa', getattr(args, 'local_spatial', False)),
                    ffn_chunk_size=getattr(args, 'ffn_chunk_size', 65536),
                    ipa_args=ipa_args if args.interleave_ipa else None,
                    local_spatial=getattr(args, 'local_spatial', False),
                    attention_backend=getattr(args, 'spatial_attention_backend', 'sdpa'),
                    group_batch_size=getattr(args, 'group_batch_size', 4),
                    cell_summaries_enabled=getattr(args, 'cell_summaries', True),
                )
                for _ in range(args.num_layers)
            ]
        )

        if not (self.args.dynamic_mpnn or self.args.mpnn):
            self.emb_to_latent = FinalLayer(args.embed_dim, latent_dim)
        if args.design:
            self.fc1 = nn.Linear(args.embed_dim, args.embed_dim)
            self.fc2 = nn.Linear(args.embed_dim, args.embed_dim)
            self.fc3 = nn.Linear(args.embed_dim, args.embed_dim)
            self.emb_to_logits = nn.Linear(args.embed_dim, 20)

        self.t_embedder = TimestepEmbedder(args.embed_dim)
        if args.abs_pos_emb:
            self.register_buffer('pos_embed',
                                 nn.Parameter(torch.zeros(1, args.crop, args.embed_dim), requires_grad=False))

        if args.abs_time_emb:
            self.register_buffer('time_embed',
                                 nn.Parameter(torch.zeros(1, args.num_frames, args.embed_dim), requires_grad=False))

        self.args = args
        if self.args.design:
            self.condflow = DirichletConditionalFlow(K=20, alpha_spacing=0.001,
                                                     alpha_max=args.alpha_max)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self.args.prepend_ipa:
            for block in self.ipa_layers:
                nn.init.constant_(block.ipa.linear_out.weight, 0)
                nn.init.constant_(block.ipa.linear_out.bias, 0)

        if self.args.interleave_ipa:
            for block in self.layers:
                nn.init.constant_(block.ipa.linear_out.weight, 0)
                nn.init.constant_(block.ipa.linear_out.bias, 0)

        # # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.args.abs_pos_emb:
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(self.args.crop))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.args.abs_time_emb:
            time_embed = get_1d_sincos_pos_embed_from_grid(self.time_embed.shape[-1], np.arange(self.args.num_frames))
            self.time_embed.data.copy_(torch.from_numpy(time_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0) #訓練開始相讓gate 唯0 讓整個map 都是0
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        if not (self.args.dynamic_mpnn or self.args.mpnn):
            # Zero-out output layers:
            nn.init.constant_(self.emb_to_latent.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.emb_to_latent.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.emb_to_latent.linear.weight, 0)
            nn.init.constant_(self.emb_to_latent.linear.bias, 0)

    def run_ipa(
            self,
            t,
            mask,
            start_frames,
            end_frames,
            aatype,
            x_d=None,
            partitions=None,
    ):
        if self.args.sim_condition or self.args.mpnn:
            B, L = mask.shape
            x = torch.zeros(B, L, self.args.embed_dim, device=mask.device)
            if aatype is not None and not self.args.no_aa_emb:
                x = x + self.aatype_to_emb(aatype)
            if self.args.design:
                x = x + self.x_d_to_emb(x_d)  # pass in only the simplex data
            for layer in self.ipa_layers:
                x = layer(x, t, mask, frames=start_frames, partitions=partitions)
        elif self.args.tps_condition or self.args.inpainting or self.args.dynamic_mpnn:
            x_f = start_frames.invert().compose(end_frames).to_tensor_7()
            x_r = end_frames.invert().compose(start_frames).to_tensor_7()
            x_f = self.latent_to_emb_f(x_f)
            x_r = self.latent_to_emb_r(x_r)
            if aatype is not None and not self.args.no_aa_emb:
                x_f = x_f + self.aatype_to_emb(aatype)
                x_r = x_r + self.aatype_to_emb(aatype)
            if self.args.design:
                x_f = x_f + self.x_d_to_emb(x_d)
                x_r = x_r + self.x_d_to_emb(x_d)
            for layer in self.ipa_layers:
                x_r = layer(x_r, t, mask, frames=start_frames, partitions=partitions)
                x_f = layer(x_f, t, mask, frames=end_frames, partitions=partitions)
            x = (x_r + x_f)

        # x = x[:, None] + x_latent
        return x

    def forward(self, x, t, mask,
                start_frames=None, end_frames=None,
                x_cond=None, x_cond_mask=None,
                aatype=None,
                chain_id=None,
                residue_index=None,
                spatial_partitions=None,
                ):
        if self.args.dynamic_mpnn:
            x = x[:, [0, -1]]
            x_cond = x_cond[:, [0, -1]]
            x_cond_mask = x_cond_mask[:, [0, -1]]
            mask = mask[:, [0, -1]]
        if self.args.mpnn:
            x = x[:, :1]
            x_cond = x_cond[:, :1]
            x_cond_mask = x_cond_mask[:, :1]
            mask = mask[:, :1]

        if self.args.design:
            x_d = x[..., -20:].mean(1)
        else:
            x_d = None

        x = self.latent_to_emb(x)  # 384 dim token
        if self.args.abs_pos_emb:
            x = x + self.pos_embed

        if self.args.abs_time_emb:
            x = x + self.time_embed[:, :, None]

        if x_cond is not None:
            x = x + self.cond_to_emb(x_cond) + self.mask_to_emb(x_cond_mask)  # token has cond g, tau

        t = self.t_embedder(t * self.args.time_multiplier)[:, None]

        partitions = spatial_partitions
        if getattr(self.args, 'local_spatial', False):
            if start_frames is None:
                raise ValueError('Local spatial execution requires conditioned start_frames')
            B, _, L = mask.shape
            if chain_id is None:
                chain_id = torch.zeros((B, L), dtype=torch.long, device=mask.device)
            if residue_index is None:
                residue_index = torch.arange(L, device=mask.device)[None].expand(B, -1)
            if partitions is None:
                partitions = [
                    build_spatial_partition(
                        start_frames.get_trans()[batch_index],
                        mask=mask[batch_index, 0].bool(),
                        chain_id=chain_id[batch_index],
                        residue_index=residue_index[batch_index],
                        cell_width=getattr(self.args, 'spatial_cell_width', 16.0),
                        contact_radius=getattr(self.args, 'spatial_contact_radius', 16.0),
                        core_capacity=getattr(self.args, 'core_capacity', 128),
                        ghost_capacity=getattr(self.args, 'ghost_capacity', 128),
                    )
                    for batch_index in range(B)
                ]

        if self.args.prepend_ipa:  # IPA doesn't need checkpointing
            x = x + self.run_ipa(
                t[:, 0], mask[:, 0], start_frames, end_frames, aatype,
                x_d=x_d, partitions=partitions,
            )[:, None]

        for layer_idx, layer in enumerate(self.layers):
            if partitions is None:
                x = grad_checkpoint(layer, (x, t, mask, start_frames), self.args.grad_checkpointing)
            else:
                x = layer(x, t, mask, start_frames, partitions=partitions)

        if not (self.args.dynamic_mpnn or self.args.mpnn):
            latent = self.emb_to_latent(x, t)
        if self.args.design:  ### this is also kind of weird
            x_l = self.fc2(gelu(self.fc1(x)))
            x_l = x_l.mean(1)
            logits = self.emb_to_logits(gelu(self.fc3(x_l)))
            if self.args.dynamic_mpnn or self.args.mpnn:
                return logits[:, None, :]
            latent[:, :, :, -20:] = latent[:, :, :, -20:] + logits[:, None, :, :]
        return latent

    # x, t, mask, start_frames=None, end_frames=None, x_cond=None, x_cond_mask=None, aatype=None
    def forward_inference(self, x, t, mask,
                          start_frames=None, end_frames=None,
                          x_cond=None, x_cond_mask=None,
                          aatype=None,
                          chain_id=None,
                          residue_index=None,
                          spatial_partitions=None,
                          ):
        if not self.args.design or self.args.dynamic_mpnn or self.args.mpnn:
            return self.forward(
                x, t, mask, start_frames, end_frames, x_cond, x_cond_mask,
                aatype, chain_id, residue_index, spatial_partitions,
            )
        else:
            x_discrete = x[:, :, :, -20:]
            B, T, L, _ = x_discrete.shape
            if not torch.allclose(x_discrete.sum(3), torch.ones((B, T, L), device=x.device), atol=1e-4) or not (
                    x_discrete >= 0).all():
                print(
                    f'WARNING: xt.min(): {x_discrete.min()}. Some values of xt do not lie on the simplex. There are '
                    f'we are '
                    f'{(x_discrete < 0).sum()} negative values in xt of shape {x_discrete.shape} that are negative. '
                    f'We are projecting '
                    f'them onto the simplex.')

                # x_discrete = simplex_proj(x_discrete)
            latent = self.forward(
                x, t, mask, start_frames, end_frames, x_cond, x_cond_mask,
                aatype, chain_id, residue_index, spatial_partitions,
            )
            latent_continuous = latent[:, :, :, :-20]
            logits = latent[:, :, :, -20:]

            flow_probs = torch.nn.functional.softmax(logits / self.args.dirichlet_flow_temp, -1)
            if not torch.allclose(flow_probs.sum(3), torch.ones((B, T, L), device=x.device), atol=1e-4) or not (
                    flow_probs >= 0).all():
                print(
                    f'WARNING: flow_probs.min(): {flow_probs.min()}. Some values of flow_probs do not lie on the '
                    f'simplex. There are we are {(flow_probs < 0).sum()} negative values in flow_probs of shape '
                    f'{flow_probs.shape} that are negative. We are projecting them onto the simplex.')
                flow_probs = simplex_proj(flow_probs)

            alpha, dalpha_dt = t_to_alpha(t[0], self.args);
            alpha = alpha.item()

            if alpha > self.args.alpha_max:
                alpha = self.args.alpha_max - self.condflow.alpha_spacing
            c_factor = self.condflow.c_factor(x_discrete.cpu().numpy(), alpha)
            c_factor = torch.from_numpy(c_factor).to(x_discrete)
            if torch.isnan(c_factor).any():
                print(f'NAN cfactor after: xt.min(): {x_discrete.min()}, flow_probs.min(): {flow_probs.min()}')

                if self.args.allow_nan_cfactor:
                    c_factor = torch.nan_to_num(c_factor)
                else:
                    raise RuntimeError(
                        f'NAN cfactor after: xt.min(): {x_discrete.min()}, flow_probs.min(): {flow_probs.min()}')

            if not (flow_probs >= 0).all(): print(f'flow_probs.min(): {flow_probs.min()}')
            eye = torch.eye(20).to(x_discrete)
            cond_flows = (eye - x_discrete.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1) * dalpha_dt

            return torch.cat([latent_continuous, flow], -1)


class AttentionWithRoPE(nn.Module):
    def __init__(self, *args, use_sdpa=False, **kwargs):
        super().__init__()
        self.use_sdpa = use_sdpa
        self.attn = MultiheadAttention(*args, **kwargs)

    def forward(self, x, mask):
        if self.use_sdpa:
            return run_pretrained_sdpa_attention(self, x, mask)
        x = x.transpose(0, 1)
        padding_mask = None if mask is None else ~mask.bool()
        x, _ = self.attn(query=x, key=x, value=x, key_padding_mask=padding_mask)
        x = x.transpose(0, 1)
        return x


class IPALayer(nn.Module):
    """Transformer layer block."""

    def __init__(self, embed_dim, ffn_embed_dim, mha_heads, dropout=0.0,
                 use_rotary_embeddings=False, ipa_args=None, local_spatial=False,
                 attention_backend='sdpa', group_batch_size=4, cell_summaries_enabled=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.mha_heads = mha_heads
        self.inf = 1e5
        self.use_rotary_embeddings = use_rotary_embeddings
        self.local_spatial = local_spatial
        self.attention_backend = attention_backend
        self.group_batch_size = group_batch_size
        self.cell_summaries_enabled = cell_summaries_enabled
        self._init_submodules(add_bias_kv=True, dropout=dropout, ipa_args=ipa_args)

    def _init_submodules(self, add_bias_kv=False, dropout=0.0, ipa_args=None):
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.embed_dim, 6 * self.embed_dim, bias=True)
        )

        self.ipa_norm = nn.LayerNorm(self.embed_dim)
        self.ipa = InvariantPointAttention(**ipa_args)

        self.mha_l = AttentionWithRoPE(
            self.embed_dim,
            self.mha_heads,
            add_bias_kv=add_bias_kv,
            dropout=dropout,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )
        if self.local_spatial:
            self.cell_summaries = CoreCellSummaryExchange(self.embed_dim, self.mha_heads)

        self.mha_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, t, mask=None, frames=None, partitions=None):
        shift_msa_l, scale_msa_l, gate_msa_l, \
            shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=-1)
        if partitions is None:
            x = x + self.ipa(self.ipa_norm(x), frames, frame_mask=mask)
        else:
            ipa_updates = []
            for batch_index, partition in enumerate(partitions):
                ipa_updates.append(
                    run_local_ipa(
                        self.ipa,
                        self.ipa_norm(x[batch_index:batch_index + 1]),
                        frames[batch_index:batch_index + 1],
                        partition,
                        group_batch_size=self.group_batch_size,
                    )
                )
            x = x + torch.cat(ipa_updates, dim=0)

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_l, scale_msa_l)
        if partitions is None:
            x = self.mha_l(x, mask=mask)
        else:
            x = torch.cat([
                run_pretrained_local_attention(
                    self.mha_l, x[batch_index:batch_index + 1], partition,
                    mask=None if mask is None else mask[batch_index:batch_index + 1],
                    backend=self.attention_backend,
                    group_batch_size=self.group_batch_size,
                )
                for batch_index, partition in enumerate(partitions)
            ], dim=0)
        x = residual + gate_msa_l.unsqueeze(1) * x
        if partitions is not None and self.cell_summaries_enabled:
            x = torch.cat([
                self.cell_summaries(x[batch_index:batch_index + 1], partition)
                for batch_index, partition in enumerate(partitions)
            ], dim=0)

        residual = x
        x = modulate(self.final_layer_norm(x), shift_mlp, scale_mlp)
        x = self.fc2(gelu(self.fc1(x)))
        x = residual + gate_mlp.unsqueeze(1) * x

        return x


class LatentMDGenLayer(nn.Module):
    """time, space Attention + adaLN """

    def __init__(self, embed_dim, ffn_embed_dim, mha_heads, dropout=0.0, num_frames=50, hyena=False,
                 use_rotary_embeddings=False, use_time_attention=True, ipa_args=None, temporal_sdpa=False,
                 local_spatial=False, attention_backend='sdpa', group_batch_size=4, cell_summaries_enabled=True,
                 ffn_chunk_size=65536):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.hyena = hyena
        self.ffn_embed_dim = ffn_embed_dim
        self.mha_heads = mha_heads
        self.inf = 1e5
        self.use_time_attention = use_time_attention
        self.temporal_sdpa = temporal_sdpa
        if ffn_chunk_size <= 0:
            raise ValueError('ffn_chunk_size must be positive')
        self.ffn_chunk_size = ffn_chunk_size
        self.use_rotary_embeddings = use_rotary_embeddings
        self.local_spatial = local_spatial
        self.attention_backend = attention_backend
        self.group_batch_size = group_batch_size
        self.cell_summaries_enabled = cell_summaries_enabled
        self._init_submodules(add_bias_kv=True, dropout=dropout, ipa_args=ipa_args)

    def _init_submodules(self, add_bias_kv=False, dropout=0.0, ipa_args=None):

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.embed_dim, 9 * self.embed_dim, bias=True)
        )

        if ipa_args is not None:
            self.ipa_norm = nn.LayerNorm(self.embed_dim)
            self.ipa = InvariantPointAttention(**ipa_args)

        if self.hyena:
            self.mha_t = HyenaOperator(
                d_model=self.embed_dim,
                l_max=self.num_frames,
                order=2,
                filter_order=64,
            )

        else:
            self.mha_t = AttentionWithRoPE(
                self.embed_dim,
                self.mha_heads,
                use_sdpa=self.temporal_sdpa,
                add_bias_kv=add_bias_kv,
                dropout=dropout,
                use_rotary_embeddings=self.use_rotary_embeddings,
            )

        self.mha_l = AttentionWithRoPE(
            self.embed_dim,
            self.mha_heads,
            add_bias_kv=add_bias_kv,
            dropout=dropout,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )
        if self.local_spatial:
            self.cell_summaries = CoreCellSummaryExchange(self.embed_dim, self.mha_heads)

        self.mha_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

    def _ffn(self, x):
        token_count = x.numel() // self.embed_dim
        if self.training or torch.is_grad_enabled() or token_count <= self.ffn_chunk_size:
            return self.fc2(gelu(self.fc1(x)))

        # Inference tokens are independent; reuse their buffer while bounding
        # the expanded FFN workspace. Grad-enabled calls retain the full path.
        if x.is_contiguous():
            tokens = x.view(-1, self.embed_dim)
            for start in range(0, token_count, self.ffn_chunk_size):
                end = min(start + self.ffn_chunk_size, token_count)
                token_chunk = tokens[start:end]
                token_chunk.copy_(self.fc2(gelu(self.fc1(token_chunk))))
        else:
            tokens_per_residue = token_count // x.shape[-2]
            residue_chunk_size = max(1, self.ffn_chunk_size // max(tokens_per_residue, 1))
            for start in range(0, x.shape[-2], residue_chunk_size):
                end = min(start + residue_chunk_size, x.shape[-2])
                token_chunk = x[..., start:end, :]
                token_chunk.copy_(self.fc2(gelu(self.fc1(token_chunk))))
        return x

    def forward(self, x, t, mask=None, frames=None, partitions=None):
        B, T, L, C = x.shape

        # msa_l : space attention ,  msa_t : time attention , mlp : feed forward network
        shift_msa_l, scale_msa_l, gate_msa_l, \
            shift_msa_t, scale_msa_t, gate_msa_t, \
            shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(9, dim=-1)

        if hasattr(self, 'ipa'):
            if partitions is None:
                x = x + self.ipa(self.ipa_norm(x), frames[:, None], frame_mask=mask)
            else:
                ipa_updates = []
                for batch_index, partition in enumerate(partitions):
                    local_frames = frames[batch_index:batch_index + 1].to_tensor_7()
                    local_frames = Rigid.from_tensor_7(
                        local_frames[:, None].expand(-1, T, -1, -1).reshape(T, L, 7)
                    )
                    ipa_updates.append(
                        run_local_ipa(
                            self.ipa,
                            self.ipa_norm(x[batch_index]),
                            local_frames,
                            partition,
                            group_batch_size=self.group_batch_size,
                        )
                    )
                x = x + torch.stack(ipa_updates, dim=0)

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_l, scale_msa_l)
        if partitions is None:
            x = self.mha_l(
                x.reshape(B * T, L, C),
                mask=mask.reshape(B * T, L),
            ).reshape(B, T, L, C)
        else:
            x = torch.stack([
                run_pretrained_local_attention(
                    self.mha_l, x[batch_index], partition,
                    mask=None if mask is None else mask[batch_index],
                    backend=self.attention_backend,
                    group_batch_size=self.group_batch_size,
                )
                for batch_index, partition in enumerate(partitions)
            ], dim=0)
        x = residual + gate_msa_l.unsqueeze(1) * x
        if partitions is not None and self.cell_summaries_enabled:
            x = torch.stack([
                self.cell_summaries(x[batch_index], partition)
                for batch_index, partition in enumerate(partitions)
            ], dim=0)

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_msa_t, scale_msa_t)
        if self.hyena:
            assert (mask - 1).sum() == 0
            x = self.mha_t(
                x.transpose(1, 2).reshape(B * L, T, C)
            ).reshape(B, L, T, C).transpose(1, 2)
        else:
            x = self.mha_t(
                x.transpose(1, 2).reshape(B * L, T, C),
                mask=mask.transpose(1, 2).reshape(B * L, T)
            ).reshape(B, L, T, C).transpose(1, 2)
        x = residual + gate_msa_t.unsqueeze(1) * x

        residual = x
        x = modulate(self.final_layer_norm(x), shift_mlp, scale_mlp)
        x = self._ffn(x)
        x = residual + gate_mlp.unsqueeze(1) * x

        return x
