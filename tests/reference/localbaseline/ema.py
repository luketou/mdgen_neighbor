"""
【檔案功能說明】
本檔案 (ema.py) 實現了指數移動平均 (Exponential Moving Average, EMA) 類別。
在深度學習模型（尤其是生成模型如 Diffusion / Flow Matching）訓練期間，使用 EMA 保存模型參數的滑動平均值，
可以有效降低權重震盪、提升模型生成品質與泛化能力。
"""

# https://github.com/aqlaboratory/openfold/blob/main/openfold/utils/exponential_moving_average.py
from collections import OrderedDict
import torch
import torch.nn as nn

from .tensor_utils import tensor_tree_map


class ExponentialMovingAverage:
    """
    【類別說明】
    指數移動平均 (EMA) 追蹤器。

    在訓練的每個步驟中，按照以下公式更新儲存的模型權重複本 `copy`:
        `copy = decay * copy + (1 - decay) * param`
    其中 `decay` (通常設為 0.999 或 0.9999) 控制舊參數的保留比例。
    """

    def __init__(self, model: nn.Module, decay: float):
        """
        【初始化函式】
        建立 EMA 追蹤器並備份當前模型的初始參數。

        參數：
            model (nn.Module): 要追蹤參數的模型。
            decay (float): 指數衰減率 (例如 0.999)。
        """
        super(ExponentialMovingAverage, self).__init__()

        clone_param = lambda t: t.clone().detach()
        self.params = tensor_tree_map(clone_param, model.state_dict())
        self.decay = decay
        self.device = next(model.parameters()).device

    def to(self, device):
        """
        【函式說明】
        將 EMA 儲存的所有參數移至指定的計算設備 (CPU/GPU)。

        參數：
            device (torch.device 或 str): 目標計算設備。
        """
        self.params = tensor_tree_map(lambda t: t.to(device), self.params)
        self.device = device

    def _update_state_dict_(self, update, state_dict):
        """
        【內部私有函式】
        遞迴更新模型參數狀態字典 (state_dict) 的 EMA 滑動平均值。

        參數：
            update (dict): 當前模型的權重字典。
            state_dict (dict): 儲存 EMA 累計權重的字典。
        """
        with torch.no_grad():
            for k, v in update.items():
                stored = state_dict[k]
                if not isinstance(v, torch.Tensor):
                    self._update_state_dict_(v, stored)
                else:
                    diff = stored - v
                    diff *= 1 - self.decay
                    stored -= diff

    def update(self, model: torch.nn.Module) -> None:
        """
        【函式說明】
        根據傳入模型當前的最新權重，計算並更新 EMA 移動平均值。

        參數：
            model (torch.nn.Module): 最新訓練步數的模型。
        """
        self._update_state_dict_(model.state_dict(), self.params)

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        """
        【函式說明】
        從保存的 checkpoint 載入 EMA 狀態（參數與 decay 值）。

        參數：
            state_dict (OrderedDict): 包含 "params" 與 "decay" 的狀態字典。
        """
        for k in state_dict["params"].keys():
            self.params[k] = state_dict["params"][k].clone()
        self.decay = state_dict["decay"]

    def state_dict(self) -> OrderedDict:
        """
        【函式說明】
        回傳 EMA 當前儲存的狀態字典，用於模型的儲存與 Checkpointing。

        回傳：
            OrderedDict: 包含 "params" 與 "decay" 的字典。
        """
        return OrderedDict(
            {
                "params": self.params,
                "decay": self.decay,
            }
        )
        super(ExponentialMovingAverage, self).__init__()

        clone_param = lambda t: t.clone().detach()
        self.params = tensor_tree_map(clone_param, model.state_dict())
        self.decay = decay
        self.device = next(model.parameters()).device

    def to(self, device):
        self.params = tensor_tree_map(lambda t: t.to(device), self.params)
        self.device = device

    def _update_state_dict_(self, update, state_dict):
        with torch.no_grad():
            for k, v in update.items():
                stored = state_dict[k]
                if not isinstance(v, torch.Tensor):
                    self._update_state_dict_(v, stored)
                else:
                    diff = stored - v
                    diff *= 1 - self.decay
                    stored -= diff

    def update(self, model: torch.nn.Module) -> None:
        """
        Updates the stored parameters using the state dict of the provided
        module. The module should have the same structure as that used to
        initialize the ExponentialMovingAverage object.
        """
        self._update_state_dict_(model.state_dict(), self.params)

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        for k in state_dict["params"].keys():
            self.params[k] = state_dict["params"][k].clone()
        self.decay = state_dict["decay"]

    def state_dict(self) -> OrderedDict:
        return OrderedDict(
            {
                "params": self.params,
                "decay": self.decay,
            }
        )