# Ticket 2：成對訓練數值門檻查核

查核日期：2026-10-23（Asia/Taipei）。狀態：供下一輪決策；本文件不是核准契約。使用者已同意維持 896 預算、在可行子集比較並揭露覆蓋率，以及先查核方法依據。既有 loss 1%、gradient／delta cosine 0.99、norm 差 5% 與 near-zero `1e-12` 仍是待決工程提案。

## 可支持的結論

本次查核的 MDGen 原文、官方程式、Adam 原論文與 PyTorch 方法文件，支持量測 masked loss、梯度方向／大小與實際 optimizer 更新；**未提供可直接移植到本專案完整蛋白 local 對 Training region 的上述指定門檻**。這是對下列查核範圍的結論，不代表所有文獻都不存在此類證據。來源也不能把一次成對數值接近轉成訓練收斂或生成品質保證。

**建議（尚待使用者決定）**：真正截斷的 case 先採描述性 loss／gradient／delta 報告，不稱為「訓練近似通過」；有限性、相同 supervision 與完整涵蓋控制另作有效性／工具一致性檢查。若仍需硬門檻，可由使用者事先指定工程容差，但必須標示未校準到下游品質。

## 原始來源究竟支持甚麼

| 來源事實 | 對 Ticket 2 的解讀與限制 |
| --- | --- |
| MDGen 原文 B.3 的蛋白實驗使用 250 幀、至多 256 殘基隨機 crop 訓練，推論時生成完整蛋白，並比較生成 ensemble 與 MD reference。[原文 B.3](https://arxiv.org/html/2409.17808v1#A2.SS3) | crop 可作為一種訓練設計的先例；原文沒有驗證本專案的空間 loss group／halo 演算法，也沒有把 crop 梯度與完整蛋白梯度以 0.99／5% 判定等價。這個先例不能直接替代本次成對比較。 |
| 官方 `transport.py` 的 velocity loss 是 masked squared error 的加總除以 mask 加總，`sample()` 按輸入 shape 抽取 noise／t。[官方 transport](https://raw.githubusercontent.com/bjing2016/mdgen/master/mdgen/transport/transport.py) | 可支持量測定義。**本專案推論**：兩側必須使用相同監督分量、相同分母，並先抽完整 noise 再按全蛋白索引選取 region；僅重設 seed、對不同 shape 各抽一次，不能保證 residue 對應的 noise 相同。 |
| 官方 wrapper 可選 Adam／AdamW，並傳入 learning rate。[官方 wrapper](https://raw.githubusercontent.com/bjing2016/mdgen/master/mdgen/wrapper.py) | 這支持記錄 optimizer 類型與實際參數；不提供 approximation acceptance threshold。AdamW 不能未註明地代替本提案的零 weight decay Adam。 |
| Adam Algorithm 1 保存梯度的一、二階動差並做 bias correction；原論文的理論討論置於 online convex framework。[Adam 原論文，Algorithm 1 與 §4](https://arxiv.org/pdf/1412.6980) | 來源支持更新方程；不支持用某個單步 delta cosine 證明本非凸模型後續訓練等價。 |
| PyTorch cosine 文件定義點積除以兩個各自受 eps 下限限制的 norm。[PyTorch 2.11 cosine](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.cosine_similarity.html) | 這是計算規則；預設 eps 是防除零設定，不是近零科學判準，更不是 cosine 0.99 的品質依據。 |
| 相對不精確梯度研究分析的是在每步 gradient error norm 有相對界限的 smooth convex 固定步長 gradient descent。[Vernimmen 與 Glineur，摘要](https://arxiv.org/abs/2506.17145v2) | 支持量測「差向量 norm／reference norm」的意義；本專案沒有確認其凸性、smoothness 常數或逐步誤差假設，而且使用 Adam，所以不能套用該收斂結論或搬用數值。此處只查核其研究範圍，未將其定理當成專案保證。 |

## 指標與例外如何解讀

**Loss。** 在相同目標及有效分母下，同報 signed difference、absolute difference、reference loss 與相對差；相對差分母固定為 reference，不能混用兩側平均。當 reference 接近零時相對差不穩定，需另列絕對差與近零分類。這是基於上述 masked loss 定義的專案方法建議。`1e-12` 並非由原文或 FP32 dtype 自動導出；尺度、有效分量與控制組誤差尚需說明。單點 loss 相近也不約束導數：例如在同一參數點兩函數值可相等而斜率不同，故 loss 不取代 backward 比較。

**Gradient。** 對固定參數名稱／shape／拼接順序，保存每群 reference RMS、candidate RMS、difference RMS、cosine 與 norm ratio，保留每群結果及最差群組。角度與長度描述不同面向；令 `r=||g_region||/||g_full||`、`c=cos(g_full,g_region)`，則非零向量恆有 `||g_region-g_full||/||g_full|| = sqrt(1+r²-2rc)`。這是幾何恆等式而非文獻品質門檻。例如 `r=1`、`c=0.99` 仍容許約 14.1% 差向量相對 norm，不能把「cosine 0.99」讀成「gradient 誤差 1%」。公式的方向定義參照 [PyTorch cosine](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.cosine_similarity.html)。

**零與缺失。** 零向量方向沒有幾何定義；近零 reference 群應以 `cosine=null`／相對差不可評估另計，附絕對差。若 reference 正常而 candidate 為零，應明列 candidate-zero 與 norm collapse，不能以 null 默默剔除。PyTorch optimizer 對 `grad=None` 跳過該參數步驟，與零 tensor 行為不同，尤其已有動差時零梯度仍可能更新。[PyTorch Adam 的 zero_grad 說明](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html#torch.optim.Adam.zero_grad) 因此預期停用的 cell summaries 參數須按名稱列成 inactive；單側缺失或非預期雙側缺失須另報，不可補零後宣稱一致。

**Clipping 前後。** PyTorch global norm clipping 把所有梯度視為一個整體，原地縮放並回傳原 total norm。[PyTorch 2.11 clip_grad_norm_](https://docs.pytorch.org/docs/2.11/generated/torch.nn.utils.clip_grad_norm_.html) **推論**：兩側都觸發 clipping 時，post-clip total norm 可能同為 1，掩蓋原本的尺度差；共同正縮放也不修復群組內方向差。主要梯度比較必須取 clipping 前，另保存兩側 clipping norm／係數與有無觸發。Lightning 2.6.5 的 `on_before_optimizer_step` 說明確認該時點已 unscale、尚未 clip。[官方 hooks 原始碼](https://raw.githubusercontent.com/Lightning-AI/pytorch-lightning/2.6.5/src/lightning/pytorch/core/hooks.py)

**Adam 空狀態單步。** 由 Algorithm 1 推導，在全新 `m=v=0`、無 weight decay、無 AMSGrad、第一步且以實際輸入 optimizer 的梯度 `g` 計算時，理想算術下 `delta_i = -lr*g_i/(|g_i|+eps)`。當 `|g_i| >> eps`，接近 `-lr*sign(g_i)`；所以梯度大小差很多仍可能得到相似更新，較小座標的 sign 差卻會在更新比較變得明顯。此為公式推論，並非品質定理。[Adam Algorithm 1](https://arxiv.org/pdf/1412.6980) 空狀態 probe 只測重新起步，不能代表恢復訓練時已有的動差歷史；若未來研究 resumed training，兩側還需複製相同 optimizer state。[PyTorch Adam state_dict／load_state_dict](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html)

實際 FP32 參數差 `theta_after-theta_before` 還含加減的捨入效應；建議實際 delta 與兩側權重 norm 一起保存，不以理想更新方程替代觀測。此為數值方法推論，尚未對本 checkpoint 校準。

## 與現有工具的接點

本地查核基準為 `c3e8455`。`mdgen/train_step_probe.py` 目前比較 precision configurations、記錄 clipping 前 gradient，但 `gradient_cosine()` 把雙零記為 1，且 CLI 預設 0.99；這是既有工具規則，不是本次訓練門檻的外部依據。工具也尚未提供本提案完整的成對 region／full supervision 及每群實際 delta。`wrapper.py` 已有 loss-core mask；`transport.py` 的 probe time 固定只處理 t，不能替代跨 shape 的 noise 對齊。以上是程式查讀結論，尚未在此研究中實作或送 GPU 驗證。

完整涵蓋控制的用途是檢查同語意路徑是否一致；其 `atol+rtol*reference RMS` 屬工具工程規則，與真正截斷的結構近似容差分開。若控制組失敗，先判定比較無效並修工具；不能把它混入截斷誤差分布後調寬門檻。

## 下一輪需要選擇的政策

1. **建議**：截斷的 loss／gradient／delta 全採描述性；只對有限性、對齊條件與完整涵蓋控制設 gate。保留差向量與逐群結果，不發布訓練近似通過率。
2. 若改採事先工程門檻，明確核准每項數字、近零分類、單側零／缺失與最差群組規則；凍結後不得依本先導的誤差調成容易通過。

即使選擇硬門檻，允許的結論也只到「指定 checkpoint、t／noise／region、FP32、單步設定下符合本工程契約」。可行子集不能支持全部種子；單步不能支持訓練收斂、穩定性或生成 ensemble／動力學品質。後者需要獨立訓練與生成評估，其測量對象可參照 [MDGen 原文 §4、B.3](https://arxiv.org/html/2409.17808v1#S4)。

## 查核範圍與未解問題

- 已讀：MDGen 原始 v1 全文的方法／蛋白評估段落、作者官方 wrapper 與 transport；Adam Algorithm 1 及理論適用範圍；與 repo 釘選版本相同的 PyTorch 2.11 Adam／cosine／clipping 文件；Lightning 2.6.5 hooks；一篇相對 inexact-gradient 原始論文的摘要適用條件。官方 MDGen raw URL 使用 `master`，為查核當日可讀版本，非永久 commit pin。
- 本地依據：`training-policy-proposal.md`、`approved-forward.json`、`discussion.md`、`mdgen/train_step_probe.py`、`mdgen/wrapper.py`、`mdgen/transport/transport.py`。未修改這些檔案，未把研究建議寫成已核准設定。
- 尚未解決：本 checkpoint 的正常數值控制誤差、合理 near-zero 尺度、候選的實際成對差值、如何把工程誤差校準到長期訓練／生成品質。沒有做完整系統性文獻回顧或收斂證明；本次查核不足以排除其他可用研究。
