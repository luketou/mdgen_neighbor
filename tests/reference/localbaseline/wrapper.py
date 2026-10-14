"""
【檔案功能說明】
本檔案 (wrapper.py) 是 MDGen 模型訓練與推論的最上層 PyTorch Lightning 封裝模組 (`NewMDGenWrapper`)。
主要負責：
1. 整合流匹配 Transport 機制與 LatentMDGenModel 神經網路架構。
2. 根據不同任務模式 (前向模擬 sim_condition, 過渡路徑採樣 tps_condition, 上採樣 cond_interval, 動態補全/設計 inpainting/design) 建立訓練與推論的條件化遮罩 (Conditioning Masks)。
3. 在訓練階段計算 3D 剛體平移、旋轉矩陣與扭轉角的 Flow Matching 損失函數。
4. 在推論階段解算 ODE 概率流 (ODE Integrators) 生成連續分子動力學軌跡，並匯出 PDB/XTC 結構。
5. 整合 EMA (指數移動平均) 權重與 WandB / Console 評估日誌。
"""

from .ema import ExponentialMovingAverage
from .logger import get_logger
from .residue_constants import aatype_to_str_sequence

logger = get_logger(__name__)

import pytorch_lightning as pl
import torch, time, os, wandb
import numpy as np
import pandas as pd
from .rigid_utils import Rigid, Rotation
from collections import defaultdict
from functools import partial

from .model.latent_model import LatentMDGenModel
from .transport.transport import create_transport, Sampler
from .utils import get_offsets, atom14_to_pdb
from .tensor_utils import tensor_tree_map
from .geometry import frames_torsions_to_atom14, atom37_to_atom14
from .geometry_loss import peptide_geometry_loss
from .full_length.partition import build_spatial_partition


def gather_log(log, world_size):
    """
    【函式說明】
    在多 GPU 分散式訓練時，跨進程收集 (All-Gather) 所有 GPU 的日誌字典。

    參數：
        log (dict): 單個進程上的指標字典。
        world_size (int): 總進程數 (GPU 數目)。

    回傳：
        dict: 跨進程合併後的指標字典。
    """
    if world_size == 1:
        return log
    log_list = [None] * world_size
    torch.distributed.all_gather_object(log_list, log)
    log = {key: sum([l[key] for l in log_list], []) for key in log}
    return log


def get_log_mean(log):
    """
    【函式說明】
    計算日誌字典中各項指標的平均值。

    參數：
        log (dict): 指標名稱對應數值列表的字典。

    回傳：
        dict: 各項指標平均值之字典。
    """
    out = {}
    for key in log:
        try:
            out[key] = np.nanmean(log[key])
        except:
            pass
    return out


DESIGN_IDX = [1, 2]
COND_IDX = [0, 3]
DESIGN_MAP_TO_COND = [0, 0, 3, 3]


class Wrapper(pl.LightningModule):
    """
    【類別說明】
    LightningModule 基礎基類。處理 EMA 權重載入/還原與通用的訓練、驗證日誌記錄。
    """

    def __init__(self, args):
        """
        【初始化函式】
        初始化 Lightning 模組基類與日誌狀態。

        參數：
            args (argparse.Namespace): 全域配置參數。
        """
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self._log = defaultdict(list)
        self.last_log_time = time.time()
        self.iter_step = 0

    def log(self, key, data):
        """
        【函式說明】
        記錄訓練與驗證指標至內部的日誌字典中。
        依據當前階段 (stage: train 或 val) 及迭代步數分類存放。

        參數：
            key (str): 指標名稱。
            data (torch.Tensor 或 float): 要記錄的指標數值。
        """
        if isinstance(data, torch.Tensor):
            data = data.mean().item()
        log = self._log
        if self.stage == 'train' or self.args.validate:
            log["iter_" + key].append(data)
        log[self.stage + "_" + key].append(data)

    def load_ema_weights(self):
        """
        【函式說明】
        在驗證或評估前，備份當前模型訓練權重並載入指數移動平均 (EMA) 權重。
        """
        # model.state_dict() contains references to model weights rather
        # than copies. Therefore, we need to clone them before calling 
        # load_state_dict().
        logger.info('Loading EMA weights')
        clone_param = lambda t: t.detach().clone()
        self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
        self.model.load_state_dict(self.ema.state_dict()["params"])

    def restore_cached_weights(self):
        """
        【函式說明】
        在驗證完成後，將模型還原為先前備份的原始訓練權重，並清空快取。
        """
        logger.info('Restoring cached weights')
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def on_before_zero_grad(self, *args, **kwargs):
        """
        【函式說明】
        PyTorch Lightning Hook。在梯度清零前更新 EMA 權重（若啟用 EMA）。
        """
        if self.args.ema:
            self.ema.update(self.model)

    def training_step(self, batch, batch_idx):
        """
        【函式說明】
        PyTorch Lightning 單步訓練邏輯。
        確保 EMA 權重與裝置同步，並呼叫 `general_step` 執行前向流匹配計算與反向傳播損失。

        參數：
            batch (dict): 當前批次的分子軌跡數據。
            batch_idx (int): 當前批次索引。

        回傳：
            torch.Tensor: 訓練損失值。
        """
        if self.args.ema:
            if (self.ema.device != self.device):
                self.ema.to(self.device)
        return self.general_step(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        """
        【函式說明】
        PyTorch Lightning 單步驗證邏輯。
        切換為 EMA 權重（若啟用），計算驗證損失並執行額外評估 (`validation_step_extra`)。

        參數：
            batch (dict): 當前批次的驗證數據。
            batch_idx (int): 當前批次索引。
        """
        if self.args.ema:
            if (self.ema.device != self.device):
                self.ema.to(self.device)
            if (self.cached_weights is None):
                self.load_ema_weights()

        self.general_step(batch, stage='val')
        self.validation_step_extra(batch, batch_idx)
        if self.args.validate and self.iter_step % self.args.print_freq == 0:
            self.print_log()

    def validation_step_extra(self, batch, batch_idx):
        """
        【函式說明】
        驗證階段的額外處理 Hook（基類為空實作，由子類別 `NewMDGenWrapper` 覆寫執行生成與結構導出）。

        參數：
            batch (dict): 當前批次數據。
            batch_idx (int): 當前批次索引。
        """
        pass

    def on_train_epoch_end(self):
        """
        【函式說明】
        PyTorch Lightning Hook。當一個訓練 Epoch 結束時，匯總並輸出訓練階段的平均指標日誌。
        """
        self.print_log(prefix='train', save=False)

    def on_validation_epoch_end(self):
        """
        【函式說明】
        PyTorch Lightning Hook。當一個驗證 Epoch 結束時，還原快取的原始訓練權重並輸出驗證指標日誌。
        """
        if self.args.ema:
            self.restore_cached_weights()
        self.print_log(prefix='val', save=False)

    def on_before_optimizer_step(self, optimizer):
        """
        【函式說明】
        PyTorch Lightning Hook。在優化器更新權重前，定期輸出日誌並檢查參數梯度是否異常（若啟用 check_grad）。

        參數：
            optimizer (torch.optim.Optimizer): 當前優化器。
        """
        if (self.trainer.global_step + 1) % self.args.print_freq == 0:
            self.print_log()

        if self.args.check_grad:
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    logger.warning(f"Param {name} has no grad")

    def on_load_checkpoint(self, checkpoint):
        """
        【函式說明】
        PyTorch Lightning Hook。載入 Checkpoint 時還原 EMA 狀態字典。

        參數：
            checkpoint (dict): 包含權重與狀態的 Checkpoint 字典。
        """
        logger.info('Loading EMA state dict')
        if self.args.ema:
            ema = checkpoint["ema"]
            self.ema.load_state_dict(ema)

    def on_save_checkpoint(self, checkpoint):
        """
        【函式說明】
        PyTorch Lightning Hook。儲存 Checkpoint 前先暫時還原原始訓練權重，並將 EMA 狀態一併儲存。

        參數：
            checkpoint (dict): 即將寫入檔案的 Checkpoint 字典。
        """
        if self.args.ema:
            if self.cached_weights is not None:
                self.restore_cached_weights()
            checkpoint["ema"] = self.ema.state_dict()

    def print_log(self, prefix='iter', save=False, extra_logs=None):
        """
        【函式說明】
        匯總、計算平均值並輸出各項日誌指標至 Console / WandB，亦可儲存為 CSV 檔案。

        參數：
            prefix (str): 日誌前綴 ('iter', 'train', 'val')。
            save (bool): 是否將日誌儲存為 CSV 檔案。
            extra_logs (dict, optional): 需額外合併記錄的指標字典。
        """
        log = self._log
        log = {key: log[key] for key in log if f"{prefix}_" in key}
        log = gather_log(log, self.trainer.world_size)
        mean_log = get_log_mean(log)

        mean_log.update({
            'epoch': self.trainer.current_epoch,
            'trainer_step': self.trainer.global_step + int(prefix == 'iter'),
            'iter_step': self.iter_step,
            f'{prefix}_count': len(log[next(iter(log))]),

        })
        if extra_logs:
            mean_log.update(extra_logs)
        try:
            for param_group in self.optimizers().optimizer.param_groups:
                mean_log['lr'] = param_group['lr']
        except:
            pass

        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            if self.args.wandb:
                wandb.log(mean_log)
            if save:
                path = os.path.join(
                    os.environ["MODEL_DIR"],
                    f"{prefix}_{self.trainer.current_epoch}.csv"
                )
                pd.DataFrame(log).to_csv(path)
        for key in list(log.keys()):
            if f"{prefix}_" in key:
                del self._log[key]

    def configure_optimizers(self):
        """
        【函式說明】
        設定模型優化器，支援 Adam 或 AdamW，並僅針對需梯度的參數進行更新。

        回傳：
            torch.optim.Optimizer: 建立好的優化器實例。
        """
        cls = torch.optim.AdamW if self.args.adamW else torch.optim.Adam
        optimizer = cls(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
        )
        return optimizer


class NewMDGenWrapper(Wrapper):
    """
    【類別說明】
    MDGen 生成模型核心封裝類別。
    繼承自 `Wrapper`，負責初始化 `LatentMDGenModel` 網路與 `Transport` 流匹配物件，
    並定義 `general_step`（計算前向流匹配 Loss）與 `sample`（推論求解生成分子軌跡）。
    """
    def __init__(self, args):
        """
        【初始化函式】
        建立 LatentMDGenModel 模型與 Transport 流匹配採樣器，並初始化 EMA。

        參數：
            args (argparse.Namespace): 全域配置參數。
        """
        super().__init__(args)
        for key in [
            'inpainting',
            'no_torsion',
            'hyena',
            'no_aa_emb',
            'supervise_all_torsions',
            'supervise_no_torsions',
            'design_key_frames',
            'no_design_torsion',
            'cond_interval',
            'mpnn',
            'dynamic_mpnn',
            'no_offsets',
            'no_frames',
        ]:
            if not hasattr(args, key):
                setattr(args, key, False)
        # args.latent_dim = 7 if not self.args.tps_condition else 14
        latent_dim = 21 if not (self.args.tps_condition or self.args.inpainting or self.args.dynamic_mpnn) else 28
        if args.design:
            latent_dim += 20
        if args.no_frames:
            latent_dim = 111
        
        self.latent_dim = latent_dim
        self.model = LatentMDGenModel(args, latent_dim)

        self.transport = create_transport(
            args,
            args.path_type,
            args.prediction,
            None,  # args.loss_weight,
            # args.train_eps,
            # args.sample_eps,
        )  # default: velocity; 
        self.transport_sampler = Sampler(self.transport)

        if not hasattr(args, 'ema'):
            args.ema = False
        if args.ema:
            self.ema = ExponentialMovingAverage(
                model=self.model, decay=args.ema_decay
            )
            self.cached_weights = None

    def prep_hyena_batch(self, batch):
        """
        【函式說明】
        針對已包含 latent 特徵的批次（例如搭配 Hyena 架構）進行前處理。
        建立前向模擬、過渡路徑採樣 (TPS)、上採樣與動態補全 (Inpainting) 所需的條件化遮罩 (cond_mask)。

        參數：
            batch (dict): 資料集載入的原始字典，包含 'latents', 'trans', 'rots', 'seqres', 'mask' 等。

        回傳：
            dict: 包含整理後 'latents', 'loss_mask' 以及神經網路輸入 'model_kwargs' 的字典。
        """
        B, T, L, _ = batch['latents'].shape
        rigids = Rigid(trans=batch['trans'], rots=Rotation(rot_mats=batch['rots']))

        ########
        cond_mask = torch.zeros(B, T, L, dtype=int, device=self.device)
        if self.args.sim_condition:
            cond_mask[:, 0] = 1
        if self.args.tps_condition:
            cond_mask[:, 0] = cond_mask[:, -1] = 1
        if self.args.cond_interval:
            cond_mask[:, ::self.args.cond_interval] = 1
        if self.args.inpainting or self.args.dynamic_mpnn or self.args.mpnn:
            cond_mask[:, :, COND_IDX] = 1

        aatype_mask = torch.ones_like(batch['seqres'])
        if self.args.design:
            aatype_mask[:, DESIGN_IDX] = 0
        ######## 
        return {
            'latents': batch['latents'].float(),
            'loss_mask': batch['loss_mask'].unsqueeze(1).expand(-1, T, -1, -1),
            'model_kwargs': {
                'start_frames': rigids,
                'mask': batch['mask'].unsqueeze(1).expand(-1, T, -1),
                'aatype': torch.where(aatype_mask.bool(), batch['seqres'], 20),
                'x_cond': torch.where(cond_mask.unsqueeze(-1).bool(), batch['latents'].float(), 0.0),
                'x_cond_mask': cond_mask,
            }
        }

    def prep_batch_no_frames(self, batch):
        """
        【函式說明】
        針對不使用剛體變換框架 (no_frames) 的模式，直接使用 atom37 坐標 (展平成 111 維) 準備訓練輸入與條件化遮罩。

        參數：
            batch (dict): 資料集載入的原始字典，包含 'atom37', 'mask', 'seqres' 等。

        回傳：
            dict: 包含 111 維 'latents', 'loss_mask' 與 'model_kwargs' 的字典。
        """
        B, T, L, _, _ = batch['atom37'].shape
        
        latents = batch['atom37'].reshape(B, T, L, 111)
        mask = batch['mask'][:,None,:,1].expand(-1, T, -1)

        loss_mask = batch['mask'][:,None,:,:,None].expand(-1, T, -1, -1, 3)
        loss_mask = loss_mask.reshape(B, T, L, 111)
        
        ########
        cond_mask = torch.zeros(B, T, L, dtype=int, device=mask.device)
        if self.args.sim_condition:
            cond_mask[:, 0] = 1
            
        aatype_mask = torch.ones_like(batch['seqres'])

        return {
            'latents': latents,
            'loss_mask': loss_mask,
            'model_kwargs': {
                'mask': mask,
                'aatype': torch.where(aatype_mask.bool(), batch['seqres'], 20),
                'x_cond': torch.where(cond_mask.unsqueeze(-1).bool(), latents, 0.0),
                'x_cond_mask': cond_mask,
            }
        }
        
    def prep_batch(self, batch):
        """
        【函式說明】
        通用數據前處理函式。將 3D 剛體（平移向量與四元數旋轉）及二面扭轉角整合成 latent 張量，
        並根據各項任務條件（sim_condition, tps_condition, cond_interval, inpainting, design 等）
        建構對應的條件化遮罩 (cond_mask)、損失遮罩 (loss_mask) 與長序列空間劃分 (spatial_partitions)。

        參數：
            batch (dict): 來自 DataLoader 的單個批次字典。

        回傳：
            dict: 包含 'rigids', 'latents', 'loss_mask' 及模型輸入參數 'model_kwargs' 的字典。
        """

        if self.args.no_frames:
            return self.prep_batch_no_frames(batch)

        # if self.args.hyena:
        if 'latents' in batch:
            return self.prep_hyena_batch(batch)

        rigids = Rigid(
            trans=batch['trans'],
            rots=Rotation(rot_mats=batch['rots'])
        )  # B, T, L
        B, T, L = rigids.shape
        if self.args.design_key_frames:
            rigids = Rigid.cat([
                rigids[:, :1, DESIGN_MAP_TO_COND],  # replace designed rototranslations in the key frames
                rigids[:, 1:-1],
                rigids[:, -1:, DESIGN_MAP_TO_COND]
            ], 1)

        if self.args.no_offsets:
            offsets = rigids.to_tensor_7()
        else:
            offsets = get_offsets(rigids[:, 0:1], rigids)
        #### make sure the quaternions have real part
        offsets[..., :4] *= torch.where(offsets[:, :, :, 0:1] < 0, -1, 1)

        frame_loss_mask = batch['mask'].unsqueeze(-1).expand(-1, -1, 7)  # B, L, 7
        torsion_loss_mask = batch['torsion_mask'].unsqueeze(-1).expand(-1, -1, -1, 2).reshape(B, L, 14)

        if self.args.tps_condition or self.args.inpainting or self.args.dynamic_mpnn:
            offsets_r = get_offsets(rigids[:, -1:], rigids)
            offsets_r[..., :4] *= torch.where(offsets_r[:, :, :, 0:1] < 0, -1, 1)
            offsets = torch.cat([offsets, offsets_r], -1)
            frame_loss_mask = torch.cat([frame_loss_mask, frame_loss_mask], -1)

        if self.args.no_torsion:
            latents = torch.cat([offsets, torch.zeros_like(batch['torsions'].view(B, T, L, 14))], -1)
        elif self.args.no_design_torsion:
            torsions_ = batch['torsions'].clone()
            torsions_[:, :, DESIGN_IDX] = 0
            latents = torch.cat([offsets, torsions_.view(B, T, L, 14)], -1)
        else:
            latents = torch.cat([offsets, batch['torsions'].view(B, T, L, 14)], -1)

        if self.args.supervise_all_torsions:
            torsion_loss_mask = torch.ones_like(torsion_loss_mask)
        elif self.args.supervise_no_torsions:
            torsion_loss_mask = torch.zeros_like(torsion_loss_mask)

        loss_mask = torch.cat([frame_loss_mask, torsion_loss_mask], -1)
        loss_mask = loss_mask.unsqueeze(1).expand(-1, T, -1, -1)

        ########
        cond_mask = torch.zeros(B, T, L, dtype=int, device=offsets.device)
        if self.args.sim_condition:
            cond_mask[:, 0] = 1
        if self.args.tps_condition:
            cond_mask[:, 0] = cond_mask[:, -1] = 1
        if self.args.cond_interval:
            cond_mask[:, ::self.args.cond_interval] = 1
        if self.args.inpainting or self.args.dynamic_mpnn or self.args.mpnn:
            cond_mask[:, :, COND_IDX] = 1

        aatype_mask = torch.ones_like(batch['seqres'])
        if self.args.design:
            aatype_mask[:, DESIGN_IDX] = 0
        ######## 

        chain_id = batch.get('chain_id')
        if chain_id is None:
            chain_id = torch.zeros_like(batch['seqres'], dtype=torch.long)
        residue_index = batch.get('residue_index')
        if residue_index is None:
            residue_index = torch.arange(L, device=offsets.device)[None].expand(B, -1)
        spatial_partitions = None
        if getattr(self.args, 'local_spatial', False):
            spatial_partitions = [
                build_spatial_partition(
                    rigids.get_trans()[batch_index, 0],
                    mask=batch['mask'][batch_index].bool(),
                    chain_id=chain_id[batch_index],
                    residue_index=residue_index[batch_index],
                    cell_width=getattr(self.args, 'spatial_cell_width', 16.0),
                    contact_radius=getattr(self.args, 'spatial_contact_radius', 16.0),
                    core_capacity=getattr(self.args, 'core_capacity', 128),
                    ghost_capacity=getattr(self.args, 'ghost_capacity', 128),
                )
                for batch_index in range(B)
            ]

        return {
            'rigids': rigids,
            'latents': latents,
            'loss_mask': loss_mask,
            'model_kwargs': {
                'start_frames': rigids[:, 0],
                'end_frames': rigids[:, -1],
                'mask': batch['mask'].unsqueeze(1).expand(-1, T, -1),
                'aatype': torch.where(aatype_mask.bool(), batch['seqres'], 20),
                'x_cond': torch.where(cond_mask.unsqueeze(-1).bool(), latents, 0.0),
                'x_cond_mask': cond_mask,
                'chain_id': chain_id,
                'residue_index': residue_index,
                'spatial_partitions': spatial_partitions,
            }
        }

    def general_step(self, batch, stage='train'):
        """
        【函式說明】
        訓練與驗證步驟的核心計算流程。
        負責調用 `prep_batch` 預處理資料，透過 `transport.training_losses` 計算 Flow Matching 損失，
        可選計算肽鏈幾何保護損失 (peptide_geometry_loss)，並記錄時間步、離散與連續損失至日誌。

        參數：
            batch (dict): 批次數據字典。
            stage (str): 執行階段 ('train' 或 'val')。

        回傳：
            torch.Tensor: 純量總損失值。
        """
        self.iter_step += 1
        self.stage = stage
        start1 = time.time()

        prep = self.prep_batch(batch)

        start = time.time()
        out_dict = self.transport.training_losses(
            model=self.model,
            x1=prep['latents'],
            aatype1=batch['seqres'] if self.args.design else None,
            mask=prep['loss_mask'],
            model_kwargs=prep['model_kwargs']
        )
        self.log('model_dur', time.time() - start)
        loss = out_dict['loss']
        self.log('loss', loss)

        cn_weight = getattr(self.args, 'cn_loss_weight', 0.0)
        ca_weight = getattr(self.args, 'ca_loss_weight', 0.0)
        if cn_weight > 0 or ca_weight > 0:
            if self.args.no_frames:
                raise ValueError('Geometry loss requires frame/torsion latents.')
            if self.args.prediction != 'velocity':
                raise ValueError('Geometry loss currently requires --prediction velocity.')

            clean_pred = out_dict['clean_pred']
            if self.args.tps_condition or self.args.inpainting:
                torsions = clean_pred[..., 14:28]
            else:
                torsions = clean_pred[..., 7:21]
            offsets = clean_pred[..., :7]

            predicted_frames = Rigid.from_tensor_7(offsets, normalize_quats=True)
            if not self.args.no_offsets:
                predicted_frames = prep['rigids'][:, 0:1].compose(predicted_frames)
            torsions = torsions.reshape(*torsions.shape[:-1], 7, 2)
            torsions = torsions / torch.linalg.vector_norm(
                torsions, dim=-1, keepdim=True
            ).clamp_min(1e-8)
            B, T, L = clean_pred.shape[:3]
            atom14 = frames_torsions_to_atom14(
                predicted_frames,
                torsions,
                batch['seqres'][:, None, :].expand(B, T, L),
            )
            geometry = peptide_geometry_loss(
                atom14,
                batch['mask'],
                residue_index=prep['model_kwargs'].get('residue_index'),
                chain_id=prep['model_kwargs'].get('chain_id'),
            )
            geometry_total = (
                cn_weight * geometry['cn_loss']
                + ca_weight * geometry['ca_loss']
            )
            loss = loss + geometry_total
            self.log('geometry_loss', geometry_total)
            for key, value in geometry.items():
                self.log(key, value.float())

        if self.args.design:
            aa_out = torch.argmax(out_dict['logits'], dim=-1)
            aa_recovery = aa_out == batch['seqres'][:, None, :].expand(-1, aa_out.shape[1], -1)

            self.log('category_pred_design_aa_recovery', aa_recovery[:, :, 1:-1].float().mean().item())
            cond_aa_recovery = torch.cat([aa_recovery[:, :, 0:1], aa_recovery[:, :, -1:]], 2)
            self.log('category_pred_cond_aa_recovery', cond_aa_recovery.float().mean().item())

            self.log('loss_continuous', out_dict['loss_continuous'].mean())
            self.log('loss_discrete', out_dict['loss_discrete'])

        self.log('time', out_dict['t'])
        self.log('dur', time.time() - self.last_log_time)
        if 'name' in batch:
            self.log('name', ','.join(batch['name']))
        self.log('general_step_dur', time.time() - start1)
        self.last_log_time = time.time()
        return loss.mean()

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

        # Dopri5 chooses its own integration steps; these are saved output
        # points. Inference consumes only the final state, so retain endpoints.
        saved_points = 2 if (self.args.sampling_method == 'dopri5'
                             and not self.training and not torch.is_grad_enabled()) else 50
        sample_fn = self.transport_sampler.sample_ode(
            sampling_method=self.args.sampling_method, num_steps=saved_points)
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

    def validation_step_extra(self, batch, batch_idx):
        """
        【函式說明】
        驗證階段的延伸評估函式。
        定期對部分批次執行完整 `inference` 軌跡採樣，計算序列設計恢復率 (AA recovery) 並將生成結構寫出為 PDB 檔案。

        參數：
            batch (dict): 當前驗證批次數據。
            batch_idx (int): 批次索引。
        """
        do_designability = batch_idx < self.args.inference_batches and (
                (self.current_epoch + 1) % self.args.designability_freq == 0 or \
                self.args.validate) and self.trainer.is_global_zero
        if do_designability:
            atom14, aa_out = self.inference(batch)
            aa_recovery = aa_out == batch['seqres'][:, None, :].expand(-1, aa_out.shape[1], -1)
            self.log('design_aa_recovery', aa_recovery[:, :, 1:-1].float().mean().item())
            cond_aa_recovery = torch.cat([aa_recovery[:, :, 0:1], aa_recovery[:, :, -1:]], 2)
            self.log('cond_aa_recovery', cond_aa_recovery.float().mean().item())
            self.log('seq_pred', ','.join([aatype_to_str_sequence(aa) for aa in aa_out[:, 0]]))
            self.log('seq_true', ','.join([aatype_to_str_sequence(aa) for aa in batch['seqres']]))
            prot_name = batch['name'][0]
            path = os.path.join(os.environ["MODEL_DIR"], f'epoch{self.current_epoch}_{prot_name}.pdb')

            atom14_to_pdb(atom14[0].cpu().numpy(), batch['seqres'][0].cpu().numpy(), path)
        else:
            self.log('design_aa_recovery', np.nan)
            self.log('cond_aa_recovery', np.nan)
            self.log('seq_pred', 'nan')
            self.log('seq_true', 'nan')

