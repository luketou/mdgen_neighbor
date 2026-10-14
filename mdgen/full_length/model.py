"""Local spatial extension of the MDGen-2026 global model.

Shares the original parameter names and temporal SDPA implementation; the
local classes only replace the global layers' spatial hooks. The layers'
fifth forward argument carries the partitions in place of position_ids.
"""
from functools import partial

import torch
from ..model.latent_model import (LatentMDGenModel as GlobalModel,
    IPALayer as GlobalIPALayer, LatentMDGenLayer as GlobalLayer)
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

    def _spatial_ipa(self, x, frames, mask, partitions):
        if partitions is None:
            return super()._spatial_ipa(x, frames, mask, None)
        return torch.cat([
            run_local_ipa(
                self.ipa,
                self.ipa_norm(x[batch_index:batch_index + 1]),
                frames[batch_index:batch_index + 1],
                partition,
                group_batch_size=self.group_batch_size,
            )
            for batch_index, partition in enumerate(partitions)
        ], dim=0)

    def _spatial_attention(self, x, mask, partitions):
        if partitions is None:
            return super()._spatial_attention(x, mask, None)
        return torch.cat([
            run_pretrained_local_attention(
                self.mha_l, x[batch_index:batch_index + 1], partition,
                mask=None if mask is None else mask[batch_index:batch_index + 1],
                backend=self.attention_backend,
                group_batch_size=self.group_batch_size,
            )
            for batch_index, partition in enumerate(partitions)
        ], dim=0)

    def _exchange_spatial(self, x, partitions):
        if partitions is None or not self.cell_summaries_enabled:
            return x
        return torch.cat([
            self.cell_summaries(x[batch_index:batch_index + 1], partition)
            for batch_index, partition in enumerate(partitions)
        ], dim=0)


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

    def _spatial_ipa(self, x, frames, mask, partitions):
        if partitions is None:
            return super()._spatial_ipa(x, frames, mask, None)
        _, T, L, _ = x.shape
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
        return torch.stack(ipa_updates, dim=0)

    def _spatial_attention(self, x, mask, partitions):
        if partitions is None:
            return super()._spatial_attention(x, mask, None)
        return torch.stack([
            run_pretrained_local_attention(
                self.mha_l, x[batch_index], partition,
                mask=None if mask is None else mask[batch_index],
                backend=self.attention_backend,
                group_batch_size=self.group_batch_size,
            )
            for batch_index, partition in enumerate(partitions)
        ], dim=0)

    def _exchange_spatial(self, x, partitions):
        if partitions is None or not self.cell_summaries_enabled:
            return x
        return torch.stack([
            self.cell_summaries(x[batch_index], partition)
            for batch_index, partition in enumerate(partitions)
        ], dim=0)


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

    def _local_partitions(self, mask, start_frames, chain_id, residue_index, partitions):
        if not self.args.local_spatial:
            return partitions
        if start_frames is None:
            raise ValueError('Local spatial execution requires conditioned start_frames')
        if partitions is None:
            partitions = partitions_from_args(
                self.args, start_frames.get_trans(), mask[:, 0],
                chain_id=chain_id, residue_index=residue_index,
            )
        return partitions

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
        partitions = self._local_partitions(
            mask, start_frames, chain_id, residue_index, spatial_partitions)
        return self._forward(
            x, t, mask, start_frames, end_frames, x_cond, x_cond_mask, aatype, partitions)

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
        return self._forward_inference(x, t, partial(
            self.forward, x, t, mask, start_frames, end_frames, x_cond, x_cond_mask,
            aatype, chain_id, residue_index, spatial_partitions,
        ))
