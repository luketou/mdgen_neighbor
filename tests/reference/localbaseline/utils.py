"""
【檔案功能說明】
本檔案 (utils.py) 包含了 MDGen 的多功能工具函式，主要負責：
1. 計算參考剛體幀 (Reference Frame) 與目標剛體之間的 7 維向量位移偏差 (`get_offsets`)。
2. Simplex 投影演算法 (`simplex_proj`) 用於序列機率分佈的 Simplex 幾何約束。
3. Dirichlet 條件流 matching 計算工具 (`DirichletConditionalFlow`)。
4. 蛋白質 Atom14/Atom37 坐標導出為標準 PDB 文件格式 (`atom14_to_pdb`, `create_full_prot`, `prots_to_pdb`)。
"""

import numpy as np
import scipy
import torch
from . import protein
from .geometry import atom14_to_atom37

def get_offsets(ref_frame, rigids):
    """
    【函式說明】
    計算參考剛體幀 (ref_frame) 與給定剛體序列 (rigids) 之間的相對 7 維變換向量 (四元數+平移)。
    考慮超長張量 (T > 500,000) 的分段處理以避免記憶體溢出。

    參數：
        ref_frame (Rigid): 參考基準剛體幀。
        rigids (Rigid): 目標剛體物件。

    回傳：
        offsets (torch.Tensor): 相對變換的 7 維張量。
    """
    B, T, L = rigids.shape
    if T > 500000:
        offsets1 = ref_frame.invert().compose(rigids[:, : 500000]).to_tensor_7()
        offsets2 = ref_frame.invert().compose(rigids[:, 500000:]).to_tensor_7()
        return torch.cat([offsets1, offsets2], 1)
    else:
        return ref_frame.invert().compose(rigids).to_tensor_7()

def simplex_proj(seq):
    """
    【函式說明】
    將向量投影回概率單純形 (Probability Simplex) 上。
    改編自 Wang & Carreira-Perpiñán (2013) 的投影演算法。

    參數：
        seq (torch.Tensor): 任意實數向量 [..., K]。

    回傳：
        X (torch.Tensor): 投影至單純形後 (總和為 1 且皆大於等於 0) 的張量。
    """
    Y = seq.reshape(-1, seq.shape[-1])
    N, K = Y.shape
    X, _ = torch.sort(Y, dim=-1, descending=True)
    X_cumsum = torch.cumsum(X, dim=-1) - 1
    div_seq = torch.arange(1, K + 1, dtype=Y.dtype, device=Y.device)
    Xtmp = X_cumsum / div_seq.unsqueeze(0)

    greater_than_Xtmp = (X > Xtmp).sum(dim=1, keepdim=True)
    row_indices = torch.arange(N, dtype=torch.long, device=Y.device).unsqueeze(1)
    selected_Xtmp = Xtmp[row_indices, greater_than_Xtmp - 1]

    X = torch.max(Y - selected_Xtmp, torch.zeros_like(Y))
    return X.view(seq.shape)

class DirichletConditionalFlow:
    """
    【類別說明】
    用於單純形 (Simplex) 上離散類別 (如 20 種胺基酸) 的 Dirichlet 條件流 Matching 計算類別。
    包含 Beta 分佈的累積分佈函數 (CDF) 數值插值計算。
    """
    def __init__(self, K=20, alpha_min=1, alpha_max=100, alpha_spacing=0.01):
        """
        【初始化函式】
        預先計算並建立 Beta CDF 的微分數值網格，用於加速後續概率流的傳輸。
        """
        self.alphas = np.arange(alpha_min, alpha_max + alpha_spacing, alpha_spacing)
        self.beta_cdfs = []
        self.bs = np.linspace(0, 1, 1000)
        for alph in self.alphas:
            self.beta_cdfs.append(scipy.special.betainc(alph, K-1, self.bs))
        self.beta_cdfs = np.array(self.beta_cdfs)
        self.beta_cdfs_derivative = np.diff(self.beta_cdfs, axis=0) / alpha_spacing
        self.alpha_spacing = alpha_spacing
        self.K = K

    def c_factor(self, bs, alpha):
        """
        【內部函式】
        計算 Dirichlet 邊界縮放因子 c_factor。
        """
        beta = scipy.special.beta(alpha, self.K - 1) # betafunction(alpha, K-1)
        beta_div = np.where(bs < 1, beta / ((1 - bs) ** (self.K - 1)), 0)
        beta_div_full = np.where((bs ** (alpha - 1)) > 0, beta_div / (bs ** (alpha - 1)), 0)

        I_func = self.beta_cdfs_derivative[np.argmin(np.abs(alpha - self.alphas))]
        interp = -np.interp(bs, self.bs, I_func)

        final = interp * beta_div_full
        return final

def atom14_to_pdb(atom14, aatype, path):
    """
    【函式說明】
    將包含多個時間幀 (Frames) 的 Atom14 蛋白質坐標張量直接轉換並寫入為標準 PDB 檔案。

    參數：
        atom14 (np.ndarray): Atom14 坐標張量 [N_frames, N_res, 14, 3]。
        aatype (np.ndarray): 殘基類型編號 [N_res]。
        path (str): 輸出的 PDB 檔案路徑。
    """
    prots = []
    for i, pos in enumerate(atom14):
        pos = atom14_to_atom37(pos, aatype)
        prots.append(create_full_prot(pos, aatype=aatype))
    with open(path, 'w') as f:
        f.write(prots_to_pdb(prots))


def create_full_prot(
        atom37: np.ndarray,
        aatype=None,
        b_factors=None,
    ):
    """
    【函式說明】
    從 Atom37 坐標陣列打包建立一個 `protein.Protein` 資料結構物件。

    參數：
        atom37 (np.ndarray): Atom37 坐標 [N_res, 37, 3]。
        aatype (np.ndarray, optional): 殘基類型。
        b_factors (np.ndarray, optional): B-factor 晶體溫控因子。
    """
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    residue_index = np.arange(n)
    atom37_mask = np.sum(np.abs(atom37), axis=-1) > 1e-7
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    chain_index = np.zeros(n, dtype=int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=chain_index
    )


def prots_to_pdb(prots):
    """
    【函式說明】
    將多幀 `protein.Protein` 物件串接合併轉換為包含 MODEL ... ENDMDL 標籤的 PDB 多幀文字格式。

    參數：
        prots (list[protein.Protein]): 蛋白質物件列表。

    回傳：
        ss (str): 多幀 PDB 格式字串。
    """
    ss = ''
    for i, prot in enumerate(prots):
        ss += f'MODEL {i}\n'
        prot = protein.to_pdb(prot)
        ss += '\n'.join(prot.split('\n')[2:-3])
        ss += '\nENDMDL\n'
    return ss

    B, T, L = rigids.shape
    if T > 500000:
        offsets1 = ref_frame.invert().compose(rigids[:, : 500000]).to_tensor_7()
        offsets2 = ref_frame.invert().compose(rigids[:, 500000:]).to_tensor_7()
        return torch.cat([offsets1, offsets2], 1)
    else:
        return ref_frame.invert().compose(rigids).to_tensor_7()

def simplex_proj(seq):
    """Algorithm from https://arxiv.org/abs/1309.1541 Weiran Wang, Miguel Á. Carreira-Perpiñán"""
    Y = seq.reshape(-1, seq.shape[-1])
    N, K = Y.shape
    X, _ = torch.sort(Y, dim=-1, descending=True)
    X_cumsum = torch.cumsum(X, dim=-1) - 1
    div_seq = torch.arange(1, K + 1, dtype=Y.dtype, device=Y.device)
    Xtmp = X_cumsum / div_seq.unsqueeze(0)

    greater_than_Xtmp = (X > Xtmp).sum(dim=1, keepdim=True)
    row_indices = torch.arange(N, dtype=torch.long, device=Y.device).unsqueeze(1)
    selected_Xtmp = Xtmp[row_indices, greater_than_Xtmp - 1]

    X = torch.max(Y - selected_Xtmp, torch.zeros_like(Y))
    return X.view(seq.shape)

class DirichletConditionalFlow:
    def __init__(self, K=20, alpha_min=1, alpha_max=100, alpha_spacing=0.01):
        self.alphas = np.arange(alpha_min, alpha_max + alpha_spacing, alpha_spacing)
        self.beta_cdfs = []
        self.bs = np.linspace(0, 1, 1000)
        for alph in self.alphas:
            self.beta_cdfs.append(scipy.special.betainc(alph, K-1, self.bs))
        self.beta_cdfs = np.array(self.beta_cdfs)
        self.beta_cdfs_derivative = np.diff(self.beta_cdfs, axis=0) / alpha_spacing
        self.alpha_spacing = alpha_spacing
        self.K = K

    def c_factor(self, bs, alpha):
        # if the bs is close to the edge of the simplex in one of its entries, then we want the c factor to be 0 for high alphas.
        # That is the rationale for why we return 0s in the case of an overflow.

        beta = scipy.special.beta(alpha, self.K - 1) # betafunction(alpha, K-1)
        beta_div = np.where(bs < 1, beta / ((1 - bs) ** (self.K - 1)), 0)
        beta_div_full = np.where((bs ** (alpha - 1)) > 0, beta_div / (bs ** (alpha - 1)), 0)

        I_func = self.beta_cdfs_derivative[np.argmin(np.abs(alpha - self.alphas))]
        interp = -np.interp(bs, self.bs, I_func)

        final = interp * beta_div_full
        return final

def atom14_to_pdb(atom14, aatype, path):
    prots = []
    for i, pos in enumerate(atom14):
        pos = atom14_to_atom37(pos, aatype)
        prots.append(create_full_prot(pos, aatype=aatype))
    with open(path, 'w') as f:
        f.write(prots_to_pdb(prots))


def create_full_prot(
        atom37: np.ndarray,
        aatype=None,
        b_factors=None,
    ):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    residue_index = np.arange(n)
    atom37_mask = np.sum(np.abs(atom37), axis=-1) > 1e-7
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    chain_index = np.zeros(n, dtype=int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        b_factors=b_factors,
        chain_index=chain_index
    )


def prots_to_pdb(prots):
    ss = ''
    for i, prot in enumerate(prots):
        ss += f'MODEL {i}\n'
        prot = protein.to_pdb(prot)
        ss += '\n'.join(prot.split('\n')[2:-3])
        ss += '\nENDMDL\n'
    return ss

