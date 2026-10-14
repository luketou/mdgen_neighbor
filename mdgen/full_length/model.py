"""Local spatial extension of the MDGen-2026 global model.

Shares the original parameter names and temporal SDPA implementation.
"""
from functools import partial

import torch
from ..model.latent_model import (LatentMDGenModel as GlobalModel,
    IPALayer as GlobalIPALayer, LatentMDGenLayer as GlobalLayer, grad_checkpoint)
from ..model.layers import gelu, modulate
from ..transport.transport import t_to_alpha
from ..utils import simplex_proj
from ..rigid_utils import Rigid
from .attention import run_pretrained_local_attention
from .local_ipa import run_local_ipa
from ..parsing import fill_local_defaults
from .partition import partitions_from_args
from .summaries import CoreCellSummaryExchange


class LocalIPALayer(GlobalIPALayer):
    def __init__(self, *args, group_batch_size=4, cell_summaries_enabled=False,
                 attention_backend='sdpa', **kwargs):
        super().__init__(*args, **kwargs)
        self.group_batch_size = group_batch_size
        self.cell_summaries_enabled = cell_summaries_enabled
        self.attention_backend = attention_backend
        self.cell_summaries = CoreCellSummaryExchange(self.embed_dim, self.mha_heads)

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


class LocalLayer(GlobalLayer):
    def __init__(self, *args, group_batch_size=4, cell_summaries_enabled=False,
                 attention_backend='sdpa', ffn_chunk_size=65536, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_batch_size = group_batch_size
        self.cell_summaries_enabled = cell_summaries_enabled
        self.attention_backend = attention_backend
        if ffn_chunk_size <= 0:
            raise ValueError('ffn_chunk_size must be positive')
        self.ffn_chunk_size = ffn_chunk_size
        self.cell_summaries = CoreCellSummaryExchange(self.embed_dim, self.mha_heads)

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


class LocalLatentMDGenModel(GlobalModel):
    def __init__(self, args, latent_dim, use_sdpa=True):
        if not use_sdpa:
            raise ValueError('local_spatial requires SDPA; manual_attention is global-only')
        if getattr(args, 'no_frames', False):
            raise ValueError('local_spatial requires residue frames')
        fill_local_defaults(args)
        super().__init__(args, latent_dim, use_sdpa=use_sdpa)
        ipa_args = dict(c_s=args.embed_dim, c_z=0, c_hidden=args.ipa_head_dim,
            no_heads=args.ipa_heads, no_qk_points=args.ipa_qk,
            no_v_points=args.ipa_v, dropout=args.dropout)
        common = dict(embed_dim=args.embed_dim, ffn_embed_dim=4*args.embed_dim,
            mha_heads=args.mha_heads, dropout=args.dropout,
            use_rotary_embeddings=not args.no_rope, use_sdpa=True,
            group_batch_size=args.group_batch_size,
            cell_summaries_enabled=args.cell_summaries,
            attention_backend=args.spatial_attention_backend)
        if args.prepend_ipa:
            self.ipa_layers = torch.nn.ModuleList([
                LocalIPALayer(**common, ipa_args=ipa_args) for _ in range(args.num_layers)])
        self.layers = torch.nn.ModuleList([
            LocalLayer(**common, num_frames=args.num_frames, hyena=args.hyena,
                use_time_attention=True,
                ipa_args=ipa_args if args.interleave_ipa else None,
                ffn_chunk_size=args.ffn_chunk_size)
            for _ in range(args.num_layers)])
        self.initialize_weights()

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
                spatial_partitions=None, position_ids=None,
                ):
        if position_ids is not None:
            raise NotImplementedError('Explicit position_ids are not supported in MDGen-2026')
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
        if self.args.local_spatial:
            if start_frames is None:
                raise ValueError('Local spatial execution requires conditioned start_frames')
            if partitions is None:
                partitions = partitions_from_args(
                    self.args, start_frames.get_trans(), mask[:, 0],
                    chain_id=chain_id, residue_index=residue_index,
                )

        if self.args.prepend_ipa:  # IPA doesn't need checkpointing
            x = x + self.run_ipa(
                t[:, 0], mask[:, 0], start_frames, end_frames, aatype,
                x_d=x_d, partitions=partitions,
            )[:, None]

        for layer_idx, layer in enumerate(self.layers):
            # Non-reentrant checkpointing accepts the non-tensor partitions.
            x = grad_checkpoint(
                partial(layer, partitions=partitions),
                (x, t, mask, start_frames),
                self.args.grad_checkpointing,
            )

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


    def forward_inference(self, x, t, mask,
                          start_frames=None, end_frames=None,
                          x_cond=None, x_cond_mask=None,
                          aatype=None,
                          chain_id=None,
                          residue_index=None,
                          spatial_partitions=None, position_ids=None,
                          ):
        if position_ids is not None:
            raise NotImplementedError('Explicit position_ids are not supported in MDGen-2026')
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
