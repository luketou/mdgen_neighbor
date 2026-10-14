"""
【檔案功能說明】
本檔案 (tensor_utils.py) 提供廣泛用於張量 (PyTorch Tensor) 操作的實用工具函式。
包含內聯加法 (inplace add)、維度轉置/展平、帶遮罩的平均值計算 (masked_mean)、
距離直方圖轉換 (distogram)、批次Gather操作，以及類似 JAX tree_map 的字典與樹狀結構映射函式 (tensor_tree_map)。
來源改編自 OpenFold / DeepMind AlphaFold2 的基礎工具庫。
"""

# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from functools import partial
from typing import List

import torch
import torch.nn as nn


def add(m1, m2, inplace):
    """
    【函式說明】
    張量加法，支援 inplace (就地) 或 out-of-place (複製) 加法。

    參數：
        m1 (torch.Tensor): 被加數張量。
        m2 (torch.Tensor): 加數張量。
        inplace (bool): 若為 True 則進行就地加法 (m1 += m2)，否則回傳新張量 (m1 + m2)。
    """
    # The first operation in a checkpoint can't be in-place, but it's
    # nice to have in-place addition during inference. Thus...
    if(not inplace):
        m1 = m1 + m2
    else:
        m1 += m2

    return m1


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    """
    【函式說明】
    僅對張量的最後幾個維度進行轉置 (Permute)，保持前面維度順序不變。

    參數：
        tensor (torch.Tensor): 輸入張量。
        inds (List[int]): 末尾維度的目標轉置順序。
    """
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    """
    【函式說明】
    將張量最後 no_dims 個維度展平 (Reshape) 合併為單一維度。

    參數：
        t (torch.Tensor): 輸入張量。
        no_dims (int): 要展平的末尾維度數量。
    """
    return t.reshape(t.shape[:-no_dims] + (-1,))


def masked_mean(mask, value, dim, eps=1e-4):
    """
    【函式說明】
    計算帶有遮罩 (Mask) 的張量平均值，自動忽略無效或填充 (Padding) 的位置。

    參數：
        mask (torch.Tensor): 遮罩張量 (1 代表有效，0 代表忽略)。
        value (torch.Tensor): 數值張量。
        dim (int): 計算平均值的維度。
        eps (float): 防止除以零的微小數值。
    """
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def pts_to_distogram(pts, min_bin=2.3125, max_bin=21.6875, no_bins=64):
    """
    【函式說明】
    將 3D 坐標點陣列計算成對距離，並離散化映射到 Distogram 距離直方圖 Bin。

    參數：
        pts (torch.Tensor): 3D 點坐標張量 [..., N, 3]。
        min_bin (float): 最小距離分箱邊界 (Å)。
        max_bin (float): 最大距離分箱邊界 (Å)。
        no_bins (int): 分箱數量。
    """
    boundaries = torch.linspace(
        min_bin, max_bin, no_bins - 1, device=pts.device
    )
    dists = torch.sqrt(
        torch.sum((pts.unsqueeze(-2) - pts.unsqueeze(-3)) ** 2, dim=-1)
    )
    return torch.bucketize(dists, boundaries)


def dict_multimap(fn, dicts):
    """
    【函式說明】
    對多個結構相同的字典列表進行映射操作，將相同鍵 (Key) 的值聚合後套用函式 `fn`。

    參數：
        fn (callable): 處理相同 Key 數值列表的對映函式。
        dicts (list[dict]): 字典列表。
    """
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


def one_hot(x, v_bins):
    """
    【函式說明】
    將連續數值 x 根據自訂的分箱邊界 v_bins 找到最近距離，並編碼為 One-Hot 向量。

    參數：
        x (torch.Tensor): 輸入張量。
        v_bins (torch.Tensor): 分箱中心點或邊界張量。
    """
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return nn.functional.one_hot(am, num_classes=len(v_bins)).float()


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    is_numpy = isinstance(data, np.ndarray)
    if is_numpy:
        data = torch.from_numpy(data)
    if isinstance(inds, np.ndarray):
        inds = torch.from_numpy(inds)

    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s, device=inds.device)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    res = data[tuple(ranges)]
    if is_numpy:
        res = res.cpu().numpy()
    return res


# With tree_map, a poor man's JAX tree_map
def dict_map(fn, dic, leaf_type):
    """
    【內部函式】
    對字典中的葉子節點進行地毯式映射。
    """
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


def tree_map(fn, tree, leaf_type):
    """
    【函式說明】
    仿照 JAX 的 `tree_map` 功能，遞迴遍歷嵌套結構 (dict, list, tuple)，對指定類型的葉子節點套用 `fn` 函式。

    參數：
        fn (callable): 作用於葉子節點的函式。
        tree (dict | list | tuple): 嵌套資料結構。
        leaf_type (type): 視為葉子的資料型別 (例如 torch.Tensor)。
    """
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        raise ValueError(f"Tree of type {type(tree)} not supported")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)