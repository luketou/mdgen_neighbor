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

`pytest --collect-only -q` 確認預設共收集 36 項測試（排除凍結參考內的附帶測試）。

另於相同 module＋venv 在 CPU 執行 `tests/test_spatial_partition.py`：6 passed。此 6 項不在 job 48275 內。另補 4 項完整 FFN／chunk=7 公開模型比較（global/local × train/eval），輸出、輸入梯度與所有權重梯度通過；此項使用 CPU，並未改動量測中的 production code。最終合計 40 項 pytest 驗證，另有 SDPA 與 checkpoint/CLI 驗證。

對應機器可讀結果在 `docs/validation/neighbor/`。固定 checkpoint SHA、GPU UUID 與輸入 SHA 記錄於 JSON；未提交大型生成座標或 checkpoint。

## 同卡大型量測：job 48276

進行中；完成後補上表格及 audit。所有結果均為合成 lattice 的工程容量驗證；不是實際大蛋白／ligand 品質驗證。

## 雙軸 review

- Standards：初審 1 個 P3，移除了無 caller 的 standalone attention、FlexAttention、backend 比較 helper 與 IPA wrapper。複核 0 未修正問題。
- Spec：程式 review 0 findings。最終大型結果與發布待核查。
