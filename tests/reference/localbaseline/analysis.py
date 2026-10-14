"""
【檔案功能說明】
本檔案 (analysis.py) 封裝了基於 PyEMMA 的蛋白質分子動力學 (MD) 分析工具函式。
包含：主鏈與側鏈扭轉角特徵抽取 (Featurizer)、TICA 降維、K-means 聚類、馬爾可夫狀態模型 (MSM) / PCCA 亞穩態分割、
過渡路徑 (Transition Path) 集群載入、轉移概率採樣 (sample_tp) 與似然值計算 (get_tp_likelihood)。
"""

import json
import os

import numpy as np
import pyemma
from tqdm import tqdm

def get_featurizer(name, sidechains=False, cossin=True):
    """
    【函式說明】
    建立 PyEMMA 特徵擷取器，提取 PDB 結構的主鏈（與可選側鏈）扭轉角。

    參數：
        name (str): PDB 檔案前綴名稱。
        sidechains (bool): 是否包含側鏈扭轉角。
        cossin (bool): 是否將角度轉為 (cos, sin) 連續向量。
    """
    feat = pyemma.coordinates.featurizer(name+'.pdb')
    feat.add_backbone_torsions(cossin=cossin)
    if sidechains:
        feat.add_sidechain_torsions(cossin=cossin)
    return feat
    
def get_featurized_traj(name, sidechains=False, cossin=True):
    """
    【函式說明】
    載入 XTC 軌跡檔並抽取扭轉角特徵。

    參數：
        name (str): Trajectory/PDB 前綴名稱。

    回傳：
        feat, traj: 特徵器物件與特徵向量軌跡。
    """
    feat = pyemma.coordinates.featurizer(name+'.pdb')
    feat.add_backbone_torsions(cossin=cossin)
    if sidechains:
        feat.add_sidechain_torsions(cossin=cossin)
    traj = pyemma.coordinates.load(name+'.xtc', features=feat)
    return feat, traj

def get_featurized_atlas_traj(name, sidechains=False, cossin=True):
    """
    【函式說明】
    載入 ATLAS 資料集專用的 XTC 軌跡並抽取特徵。
    """
    feat = pyemma.coordinates.featurizer(name+'.pdb')
    feat.add_backbone_torsions(cossin=cossin)
    if sidechains:
        feat.add_sidechain_torsions(cossin=cossin)
    traj = pyemma.coordinates.load(name+'_prod_R1_fit.xtc', features=feat)
    return feat, traj

def get_tica(traj, lag=1000):
    """
    【函式說明】
    計算 TICA (Time-lagged Independent Component Analysis) 時間滯後獨立成分分析進行動力學降維。

    參數：
        traj: 特徵化軌跡。
        lag (int): 滯後時間步數。
    """
    tica = pyemma.coordinates.tica(traj, lag=lag, kinetic_map=True)
    # lag time 100 ps = 0.1 ns
    return tica, tica.transform(traj)

def get_kmeans(traj):
    """
    【函式說明】
    對 TICA 降維後的軌跡進行 K-means 微觀狀態 (Microstates, k=100) 聚類。
    """
    kmeans = pyemma.coordinates.cluster_kmeans(traj, k=100, max_iter=100, fixed_seed=137)
    return kmeans, kmeans.transform(traj)[:,0]

def get_msm(traj, lag=1000, nstates=10):
    """
    【函式說明】
    估算馬爾可夫狀態模型 (MSM) 並進行 PCCA+ 粗粒化分割（聚合為 nstates 個亞穩態）。

    參數：
        nstates (int): 目標亞穩態數量 (預設 10)。
    """
    msm = pyemma.msm.estimate_markov_model(traj, lag=lag)
    pcca = msm.pcca(nstates)
    assert len(msm.metastable_assignments) == 100
    cmsm = pyemma.msm.estimate_markov_model(msm.metastable_assignments[traj], lag=lag)
    return msm, pcca, cmsm

def discretize(traj, kmeans, msm):
    """
    【函式說明】
    將連續軌跡轉化為 MSM 亞穩態的離散狀態序列。
    """
    return msm.metastable_assignments[kmeans.transform(traj)[:,0]]

def load_tps_ensemble(name, directory):
    """
    【函式說明】
    載入 TPS (Transition Path Sampling) 生成的軌跡系綜與 JSON 元數據。
    """
    metadata = json.load(open(os.path.join(directory, f'{name}_metadata.json'),'rb'))
    all_feats = []
    all_traj = []
    for i, meta_dict in tqdm(enumerate(metadata)):
        feats, traj = get_featurized_traj(f'{directory}/{name}_{i}', sidechains=True)
        all_feats.append(feats)
        all_traj.append(traj)
    return all_feats, all_traj


def sample_tp(trans, start_state, end_state, traj_len, n_samples):
    """
    【函式說明】
    根據轉移概率矩陣 (Transition Matrix)，以橋接條件採樣固定起點與終點的理論過渡路徑 (Transition Paths)。

    參數：
        trans (np.ndarray): MSM 狀態轉移矩陣。
        start_state (int): 起始狀態索引。
        end_state (int): 終點狀態索引。
        traj_len (int): 軌跡長度 N。
        n_samples (int): 採樣路徑數量。
    """
    s_1 = start_state
    s_N = end_state
    N = traj_len

    s_t = np.ones(n_samples, dtype=int) * s_1
    states = [s_t]
    for t in range(1, N - 1):
        numerator = np.linalg.matrix_power(trans, N - t - 1)[:, s_N] * trans[s_t, :]
        probs = numerator / np.linalg.matrix_power(trans, N - t)[s_t, s_N][:, None]
        s_t = np.zeros(n_samples, dtype=int)
        for n in range(n_samples):
            s_t[n] = np.random.choice(np.arange(len(trans)), 1, p=probs[n])
        states.append(s_t)
    states.append(np.ones(n_samples, dtype=int) * s_N)
    return np.stack(states, axis=1)


def get_tp_likelihood(tp, trans):
    """
    【函式說明】
    計算給定過渡路徑序列在理論 MSM 轉移矩陣下的條件對數似然值 (Likelihood)。
    """
    N = tp.shape[1]
    n_samples = tp.shape[0]
    s_N = tp[0, -1]
    trans_probs = []
    for i in range(N - 1):
        t = i + 1
        s_t = tp[:, i]
        numerator = np.linalg.matrix_power(trans, N - t - 1)[:, s_N] * trans[s_t, :]
        probs = numerator / np.linalg.matrix_power(trans, N - t)[s_t, s_N][:, None]

        s_tp1 = tp[:, i + 1]
        trans_prob = probs[np.arange(n_samples), s_tp1]
        trans_probs.append(trans_prob)
    probs = np.stack(trans_probs, axis=1)
    probs[np.isnan(probs)] = 0
    return probs


def get_state_probs(tp, num_states=10):
    """
    【函式說明】
    計算過渡路徑中各亞穩態出現的概率分佈。
    """
    stationary = np.bincount(tp.reshape(-1), minlength=num_states)
    return stationary / stationary.sum()

    feat = pyemma.coordinates.featurizer(name+'.pdb')
    feat.add_backbone_torsions(cossin=cossin)
    if sidechains:
        feat.add_sidechain_torsions(cossin=cossin)
    return feat
    
def get_featurized_traj(name, sidechains=False, cossin=True):
    feat = pyemma.coordinates.featurizer(name+'.pdb')
    feat.add_backbone_torsions(cossin=cossin)
    if sidechains:
        feat.add_sidechain_torsions(cossin=cossin)
    traj = pyemma.coordinates.load(name+'.xtc', features=feat)
    return feat, traj

def get_featurized_atlas_traj(name, sidechains=False, cossin=True):
    feat = pyemma.coordinates.featurizer(name+'.pdb')
    feat.add_backbone_torsions(cossin=cossin)
    if sidechains:
        feat.add_sidechain_torsions(cossin=cossin)
    traj = pyemma.coordinates.load(name+'_prod_R1_fit.xtc', features=feat)
    return feat, traj

def get_tica(traj, lag=1000):
    tica = pyemma.coordinates.tica(traj, lag=lag, kinetic_map=True)
    # lag time 100 ps = 0.1 ns
    return tica, tica.transform(traj)

def get_kmeans(traj):
    kmeans = pyemma.coordinates.cluster_kmeans(traj, k=100, max_iter=100, fixed_seed=137)
    return kmeans, kmeans.transform(traj)[:,0]

def get_msm(traj, lag=1000, nstates=10):
    msm = pyemma.msm.estimate_markov_model(traj, lag=lag)
    pcca = msm.pcca(nstates)
    assert len(msm.metastable_assignments) == 100
    cmsm = pyemma.msm.estimate_markov_model(msm.metastable_assignments[traj], lag=lag)
    return msm, pcca, cmsm

def discretize(traj, kmeans, msm):
    return msm.metastable_assignments[kmeans.transform(traj)[:,0]]

def load_tps_ensemble(name, directory):
    metadata = json.load(open(os.path.join(directory, f'{name}_metadata.json'),'rb'))
    all_feats = []
    all_traj = []
    for i, meta_dict in tqdm(enumerate(metadata)):
        feats, traj = get_featurized_traj(f'{directory}/{name}_{i}', sidechains=True)
        all_feats.append(feats)
        all_traj.append(traj)
    return all_feats, all_traj


def sample_tp(trans, start_state, end_state, traj_len, n_samples):
    s_1 = start_state
    s_N = end_state
    N = traj_len

    s_t = np.ones(n_samples, dtype=int) * s_1
    states = [s_t]
    for t in range(1, N - 1):
        numerator = np.linalg.matrix_power(trans, N - t - 1)[:, s_N] * trans[s_t, :]
        probs = numerator / np.linalg.matrix_power(trans, N - t)[s_t, s_N][:, None]
        s_t = np.zeros(n_samples, dtype=int)
        for n in range(n_samples):
            s_t[n] = np.random.choice(np.arange(len(trans)), 1, p=probs[n])
        states.append(s_t)
    states.append(np.ones(n_samples, dtype=int) * s_N)
    return np.stack(states, axis=1)


def get_tp_likelihood(tp, trans):
    N = tp.shape[1]
    n_samples = tp.shape[0]
    s_N = tp[0, -1]
    trans_probs = []
    for i in range(N - 1):
        t = i + 1
        s_t = tp[:, i]
        numerator = np.linalg.matrix_power(trans, N - t - 1)[:, s_N] * trans[s_t, :]
        probs = numerator / np.linalg.matrix_power(trans, N - t)[s_t, s_N][:, None]

        s_tp1 = tp[:, i + 1]
        trans_prob = probs[np.arange(n_samples), s_tp1]
        trans_probs.append(trans_prob)
    probs = np.stack(trans_probs, axis=1)
    probs[np.isnan(probs)] = 0
    return probs


def get_state_probs(tp, num_states=10):
    stationary = np.bincount(tp.reshape(-1), minlength=num_states)
    return stationary / stationary.sum()