"""Frozen pre-endpoint-storage inference, including its 50-point sampler default."""
import torch
from mdgen.wrapper import (NewMDGenWrapper, Rigid, Rotation, partial,
                           frames_torsions_to_atom14, atom37_to_atom14)


class FiftyPointWrapper(NewMDGenWrapper):
    def inference(self, batch):
        """
        【函式說明】
        推論與軌跡生成主函式。
        從先驗高斯分佈（或狄利克雷分佈）採樣先驗雜訊，透過反向 ODE 求解器（Euler, Midpoint, Dopri5 等）
        數值積分生成潛在軌跡向量，並將剛體與扭轉角還原重建為 3D 分子結構 (atom14/atom37)。

        參數：
            batch (dict): 輸入的條件化結構或先驗資料批次。

        回傳：
            tuple: (atom14 坐標張量, 預測之胺基酸序列類別 aa_out)。
        """
        prep = self.prep_batch(batch)

        latents = prep['latents']
        if not self.args.no_frames:
            rigids = prep['rigids']
            B, T, L = rigids.shape
        else:
            B, T, L, _ = latents.shape

        ### oracle
        # if self.args.oracle:
        #     assert self.args.sim_condition  # only works with that
        #     offsets = get_offsets(rigids[:, 0:1], rigids)
        #     torsions = batch['torsions'].view(B, T, L, 14)
        # else:
        if self.args.dynamic_mpnn or self.args.mpnn:
            x1 = prep['latents']
            x_d = torch.zeros(x1.shape[0], x1.shape[1], x1.shape[2], 20, device=self.device)
            xt = torch.cat([x1, x_d], dim=-1)
            logits = self.model.forward_inference(xt, torch.ones(B, device=self.device),
                                                  **prep['model_kwargs'])
            aa_out = torch.argmax(logits, -1)
            atom14 = frames_torsions_to_atom14(rigids, batch['torsions'],
                                               batch['seqres'][:, None].expand(B, T, L))
            return atom14, aa_out

        if self.args.design:
            zs_continuous = torch.randn(B, T, L, self.latent_dim - 20, device=latents.device)
            zs_discrete = torch.distributions.Dirichlet(torch.ones(B, L, 20, device=latents.device)).sample()
            zs_discrete = zs_discrete[:, None].expand(-1, T, -1, -1)
            zs = torch.cat([zs_continuous, zs_discrete], -1)
        else:
            zs = torch.randn(B, T, L, self.latent_dim, device=self.device)

        sample_fn = self.transport_sampler.sample_ode(sampling_method=self.args.sampling_method)
        # num_steps=self.args.inference_steps)  # default to ode

        samples = sample_fn(
            zs,
            partial(self.model.forward_inference, **prep['model_kwargs'])
        )[-1]

        if self.args.no_frames:
            atom14 = atom37_to_atom14(
                samples.cpu().numpy().reshape(B, T, L, 37, 3),
                batch['seqres'][0].cpu().numpy()
            )
            return torch.from_numpy(atom14).float(), None
            
        offsets = samples[..., :7]
        
        if self.args.tps_condition or self.args.inpainting:
            torsions = samples[..., 14:28]
            logits = samples[..., -20:]
        else:
            torsions = samples[..., 7:21]
            logits = samples[..., -20:]

        
        if self.args.no_offsets:
            frames = Rigid.from_tensor_7(offsets, normalize_quats=True)
        else:
            frames = rigids[:, 0:1].compose(Rigid.from_tensor_7(offsets, normalize_quats=True))
        if self.args.design:
            trans = frames.get_trans()
            rots = frames.get_rots().get_rot_mats()
            frames = Rigid(trans=trans, rots=Rotation(rot_mats=rots))
        torsions = torsions.reshape(B, T, L, 7, 2)
        if not self.args.oracle:
            torsions = torsions / torch.linalg.norm(torsions, dim=-1, keepdims=True)
        atom14 = frames_torsions_to_atom14(frames, torsions.view(B, T, L, 7, 2),
                                           batch['seqres'][:, None].expand(B, T, L))

        if self.args.design:
            aa_out = torch.argmax(logits, -1)
        else:
            aa_out = batch['seqres'][:, None].expand(B, T, L)
        return atom14, aa_out
