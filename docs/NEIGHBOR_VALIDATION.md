# 整合驗證紀錄（2026-10-14）

## 固定基準與 TDD

目的地為 mdgen2026，local 來源為 sibling mdgen。整合前 Git 固定點 `383d446`；雙凍結參考 `tests/reference/` 的 62 個檔案 SHA256 已逐一核對。

| 項目 | RED | GREEN |
|---|---|---|
| Local 公開模型介面 | 還原原 wrapper 後 local 回歸 1 failed、global 4 passed | 接入 local 後 5 passed；後增 train/eval × RoPE 組合 |
| ODE 起終點保存 | job 48269：原路徑 50 點，顯存保存比較 1 failed | job 48270：11 passed；正式 checkpoint 最終輸出與 NFE 相同 |
| CUDA 大批次四元數 | job 48271：32000 組旋轉矩陣觸發 cuSOLVER INVALID_VALUE | 分塊後可生成有限值座標；修正測試的序列 shape 為 [B,T,L]，job 48274 全通過 |

來源既有 core-only／逐批 gather 安全測試亦移入，未重新改寫 frozen 基準。中途失敗不是最終通過證據；最終採以下完成的 job。

## 最終回歸：job 48275

同一實體 L40S，Slurm `COMPLETED`、`ExitCode=0:0`：

- 30 項模型／local attention／ODE 測試通過。
- 原生 SDPA 的 6 組 output、gradient、padding 比較通過。
- 正式 ATLAS checkpoint，104 殘基：global 輸出、輸入梯度、權重梯度與凍結 2026 完全一致。
- Local 輸出與凍結 mdgen 完全一致；輸入梯度 max abs `1.0913936421275139e-10`，權重梯度 max abs `2.60770320892334e-8`，符合比較容差。
- ODE 保存 50／2 點：250 frames 最後 atom14 完全一致，NFE 116／116，全部 finite。
- 真實 `sim_inference.py` CLI 的 global、local 各 4 frames、1 rollout，產出 atom14／PDB／XTC，流程完成。

`pytest --collect-only -q` 確認最終預設共收集 40 項測試（排除凍結參考內的附帶測試）。

另於相同 module＋venv 在 CPU 執行 `tests/test_spatial_partition.py`：6 passed。此 6 項不在 job 48275 內。另補 4 項完整 FFN／chunk=7 公開模型比較（global/local × train/eval），輸出、輸入梯度與所有權重梯度通過；此項使用 CPU，並未改動量測中的 production code。最終合計 40 項 pytest 驗證，另有 SDPA 與 checkpoint/CLI 驗證。

對應機器可讀結果在 `docs/validation/neighbor/`。固定 checkpoint SHA、GPU UUID 與輸入 SHA 記錄於 JSON；未提交大型生成座標或 checkpoint。

## 同卡大型量測：job 48276

Slurm `COMPLETED`、`ExitCode=0:0`；運行 2:20:02。8 個案例成功、0 OOM、全部輸出 finite。量測後 code SHA256 核對一致，production 實作固定在 `5e8027c`，之後只增加測試、文件及既有檔案歸檔。

| L | 模式 | Forward 秒 | Forward allocated / reserved GiB | 生成分鐘 | 生成 allocated / reserved GiB | NFE | 結果 |
|---:|---|---:|---:|---:|---:|---:|---|
| 5000 | global | 9.13 | 20.07 / 28.15 | 20.24 | 22.63 / 31.14 | 134 | 成功 |
| 5000 | local | 11.32 | 20.09 / 21.93 | 25.52 | 22.64 / 29.40 | 134 | 成功 |
| 7500 | global | 17.42 | 30.02 / 42.15 | 49.32 | 33.87 / 41.52 | 170 | 成功 |
| 7500 | local | 17.03 | 30.04 / 32.77 | 31.36 | 33.88 / 37.55 | 110 | 成功 |

Allocated 是活躍 tensor 的峰值；reserved 是 PyTorch allocator 保留的 CUDA 記憶體，包含快取。此次 local 的 allocated 峰值與 global 接近，reserved 較低。不能把 reserved 差異視為完全相同數量的 tensor 記憶體減少。

L=5000 local 完整生成較慢；L=7500 local 完整生成較快，但 NFE 為 110、global 為 170。這包含不同空間鄰域造成的 flow field 與求解步數差異，不能全部歸因於每次 forward 的運算加速。完整生成每例只測一次，不主張統計顯著加速。

共同條件 audit 通過：同一實體 L40S UUID、checkpoint SHA、source SHA、FP32、TF32 off、250 frames、seed 137、summary off、FFN chunk 65536、dopri5、atol=1e-6、rtol=1e-3、ODE 保存 2 點。每個 L 的四個 raw input hash 一致。原始 JSON 與 CSV 位於 `docs/validation/neighbor/scaling_48276/`。

所有結果均為合成 lattice 的工程容量驗證；不是實際大蛋白／ligand 品質驗證。此次最高測到 L=7500，沒有測出系統的最大 L。

## 雙軸 review

- Standards：初審 1 個 P3，移除了無 caller 的 standalone attention、FlexAttention、backend 比較 helper 與 IPA wrapper。複核 0 未修正問題。
- Spec：最終複核 0 findings；大型原始結果、共同條件、規格範圍及凍結基準均通過。
- 最終程式與量測審查範圍為 `383d446...b4c8648`；後續僅更新審查及發布狀態文件。
- 已發布至 [luketou/mdgen_neighbor](https://github.com/luketou/mdgen_neighbor) 的 `main`。首次發布 commit `e55bc13` 的遠端 SHA 與本機一致；後续僅提交發布完成狀態。

## 發布與交接

四個 tickets 已完成；[GitHub Issue #1](https://github.com/luketou/mdgen_neighbor/issues/1) 保存規格與驗收狀態。來源歷史與授權保留，既有 `origin` 不變；發布目的地使用 `neighbor` remote。

新增發布內容包含程式、凍結參考、測試、操作文件與量測 JSON／CSV；沒有新加入 checkpoint 或生成軌跡。既有來源歷史的檔案保留，並未改寫 Git 歷史。
