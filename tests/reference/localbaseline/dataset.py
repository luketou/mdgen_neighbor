"""
【檔案功能說明】
本檔案 (dataset.py) 定義了 MDGen 的 PyTorch Dataset 類別 (`MDGenDataset`)。
負責從預處理好的 `.npy` 記憶體對映檔 (Memmap) 中載入 3D 分子軌跡片段，
並將 `atom14` 坐標陣列轉換為：
1. 剛體變換幀 (Rigid Body Frames: Translation + Rotation)
2. 7 個二面扭轉角 (Torsions: cos, sin 向量)
3. 殘基序列與 Padding/Crop 遮罩 (Mask)
供生成模型進行訓練與驗證。
"""

import torch
from .rigid_utils import Rigid
from .residue_constants import restype_order
import numpy as np
import pandas as pd
from .geometry import atom37_to_torsions, atom14_to_atom37, atom14_to_frames
       
class MDGenDataset(torch.utils.data.Dataset):
    """
    【類別說明】
    分子動力學軌跡生成訓練資料集。
    支援從記憶體對映 (.npy) 中隨機截取指定張數的軌跡幀 (Frames) 以及殘基序列截取 (Crop/Padding)。
    """
    def __init__(self, args, split, repeat=1):
        """
        【初始化函式】
        載入資料集清單 CSV 檔並設定重複次數。

        參數：
            args (argparse.Namespace): 全域命令列參數。
            split (str): Split CSV 檔案路徑 (例如 splits/4AA_train.csv)。
            repeat (int): 資料集重複載入倍率。
        """
        super().__init__()
        self.df = pd.read_csv(split, index_col='name')
        self.args = args
        self.repeat = repeat

    def __len__(self):
        """
        【函式說明】
        回傳資料集的樣本總數。
        """
        if self.args.overfit_peptide:
            return 1000
        return self.repeat * len(self.df)

    def __getitem__(self, idx):
        """
        【函式說明】
        根據索引 idx 載入並處理單個分子軌跡樣本。

        處理步驟：
        1. 開啟對應分子的 `.npy` 檔案。
        2. 隨機選取連續的 `num_frames` 個時間步。
        3. 透過 `atom14_to_frames` 計算每幀每個殘基的 3D 剛體平移向量與旋轉矩陣。
        4. 透過 `atom37_to_torsions` 計算殘基的主鏈與側鏈 7 個扭轉角。
        5. 對 ATLAS 大型蛋白質進行隨機 Crop 或 Padding 填充。

        回傳：
            dict: 包含 'trans', 'rots', 'torsions', 'seqres', 'mask' 等張量的字典。
        """
        idx = idx % len(self.df)
        idx = idx % len(self.df)
        if self.args.overfit:
            idx = 0

        if self.args.overfit_peptide is None:
            name = self.df.index[idx]
            seqres = self.df.seqres[name]
        else:
            name = self.args.overfit_peptide
            seqres = name

        if self.args.atlas:
            i = np.random.randint(1, 4)
            full_name = f"{name}_R{i}"
        else:
            full_name = name
        arr = np.lib.format.open_memmap(f'{self.args.data_dir}/{full_name}{self.args.suffix}.npy', 'r')
        if self.args.frame_interval:
            arr = arr[::self.args.frame_interval]
        
        frame_start = np.random.choice(np.arange(arr.shape[0] - self.args.num_frames))
        if self.args.overfit_frame:
            frame_start = 0
        end = frame_start + self.args.num_frames
        # arr = np.copy(arr[frame_start:end]) * 10 # convert to angstroms
        arr = np.copy(arr[frame_start:end]).astype(np.float32) # / 10.0 # convert to nm
        if self.args.copy_frames:
            arr[1:] = arr[0]

        # arr should be in ANGSTROMS
        frames = atom14_to_frames(torch.from_numpy(arr))
        seqres = np.array([restype_order[c] for c in seqres])
        aatype = torch.from_numpy(seqres)[None].expand(self.args.num_frames, -1)
        atom37 = torch.from_numpy(atom14_to_atom37(arr, aatype)).float()
        
        L = frames.shape[1]
        mask = np.ones(L, dtype=np.float32)
        
        if getattr(self.args, 'no_frames', False):
            return {
                'name': full_name,
                'frame_start': frame_start,
                'atom37': atom37,
                'seqres': seqres,
                'mask': restype_atom37_mask[seqres], # (L,)
            }
        torsions, torsion_mask = atom37_to_torsions(atom37, aatype)
        
        torsion_mask = torsion_mask[0]
        
        if self.args.atlas:
            if L > self.args.crop:
                start = np.random.randint(0, L - self.args.crop + 1)
                torsions = torsions[:,start:start+self.args.crop]
                frames = frames[:,start:start+self.args.crop]
                seqres = seqres[start:start+self.args.crop]
                mask = mask[start:start+self.args.crop]
                torsion_mask = torsion_mask[start:start+self.args.crop]
                
            
            elif L < self.args.crop:
                pad = self.args.crop - L
                frames = Rigid.cat([
                    frames, 
                    Rigid.identity((self.args.num_frames, pad), requires_grad=False, fmt='rot_mat')
                ], 1)
                mask = np.concatenate([mask, np.zeros(pad, dtype=np.float32)])
                seqres = np.concatenate([seqres, np.zeros(pad, dtype=int)])
                torsions = torch.cat([torsions, torch.zeros((torsions.shape[0], pad, 7, 2), dtype=torch.float32)], 1)
                torsion_mask = torch.cat([torsion_mask, torch.zeros((pad, 7), dtype=torch.float32)])

        return {
            'name': full_name,
            'frame_start': frame_start,
            'torsions': torsions,
            'torsion_mask': torsion_mask,
            'trans': frames._trans,
            'rots': frames._rots._rot_mats,
            'seqres': seqres,
            'mask': mask, # (L,)
        }

