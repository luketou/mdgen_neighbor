# Ticket 2：L40S 小尺寸訓練探針驗證

2026-10-23，node-201 / NVIDIA L40S。最終整合版 `61c6197` 的 Job **49247** 已由 `scontrol` 確認 `COMPLETED`、`ExitCode=0:0`，執行 2 分 2 秒。資源：1 GPU、1 CPU、2 GiB，5 分鐘上限。

[最終摘要](final/summary.json)、[最終原始報告](final/results.json)、[乾淨快照 manifest](final/manifest.json) 對應 `61c6197`，dirty=false，所有部署檔案的 SHA-256 已與送審來源核對。主線的新增 batch shape／Training region 報告功能也在此整合版中；本次仍只測未啟用 Training region 的小尺寸 batch。先前 Job 49246 是合併主線前的首次修正驗證，結果相同，原始證據亦保留於下方。

這次對實際 `NewMDGenWrapper` 執行 Lightning loss、backward、gradient clipping 與 optimizer step；不是模擬訓練。輸入為固定 seed 137、隨機初始化權重與合成 batch：batch 2、12 residues、4 frames、2 layers、embed_dim 32。global／local 各執行兩次獨立 CLI；每次比較 FP32、重複 FP32 與 BF16。

| 檢查 | global | local |
| --- | --- | --- |
| 重複 FP32 最差 cosine | 1.0 | 0.9999999999999999 |
| 重複 FP32 最大梯度絕對差 | 0 | 0 |
| BF16 對 FP32 最差 cosine | 0.999997046091613 | 0.9999975726265548 |
| 所有參數群 ≥ 0.99 | 通過 | 通過 |
| 兩次獨立 CLI 的 loss／梯度摘要完全相同 | 通過 | 通過 |
| 正常輸入梯度／step 後參數的非有限值 | 0 | 0 |

重複 FP32 的 cosine 以 1e-12 容忍度判為 1，原始最大梯度差為 0。所有訓練組態的 master weights 維持 FP32；BF16 step 內 autocast 為 bfloat16。單步時間、峰值記憶體、每群梯度與實際精度狀態均保存在原始報告。單步含暖機，不據此宣稱 BF16 加速。

另外，5 項 GPU pytest 通過，包含 global／local BF16 真實探針回歸、CUDA 峰值、NaN 偵測與 cosine 邊界。額外刻意注入 NaN／Inf，各偵測到 116045 個非有限梯度，loss 標示非有限、比較判定失敗；正常輸入則全部為有限值。

## 兩個阻塞與修正依據

1. **環境**：Job 49233 的模型匯入失敗；最小 `python -c 'import deepspeed'` 在 job 49234 連續兩次失敗（約 3–5 秒），錯誤為 `CUDA_HOME does not exist`。Job 49235 只設定已確認的 CUDA 12.8 module 路徑即可匯入成功。修正 `scripts/hpc.py` 傳遞 `runtime.env`，並將非機密 `CUDA_HOME` 納入設定；公開 CLI 回歸由失敗轉為通過。未安裝套件或修改系統 module。
2. **CUDA BF16**：Job 49240 的兩次 FP32 step 完成，BF16 在 QKV dtype assertion 失敗。Job 49244 以 8 維、2 heads、無 RoPE 的 MHA 連續重現；job 49245 測到投影與縮放均 BF16，串接 FP32 bias 後只有 K/V 變為 FP32，且 global/local 新 GPU 回歸均失敗。修正僅在 autocast SDPA 入口依實際 autocast dtype 對齊 QKV，保留 dtype assertion、FP32 參數、先前幾何及 RoPE 計算。Job 49246 原始場景及最小案例均通過。

較早 job 49228 的共用 preflight 已確認真正 CUDA matmul 通過，並曾由 `scontrol` 確認 `COMPLETED / 0:0`。歷史 GPU 裝置 EIO 在這次 preflight 沒有重現，未據此推論其修復原因。其他準備失敗的診斷快照保留供查核，不代表模型失敗：job 49239 未套用新環境、job 49242 的最小腳本缺少 package 搜尋路徑。

## 原始證據與重跑

- 首次修正驗證 Job 49246 的 [摘要與來源雜湊](summary.json)、[完整原始報告](results.json)、[快照 manifest](manifest.json)。快照基底為 `8f77bae`，當時含尚未提交的 MHA／GPU 測試修正；兩檔雜湊已與後續 `bb40f1f` 的實際內容核對一致。快照沒有 `.git`，所以探針內 `environment.git_commit` 為 null；來源由 manifest 與逐檔 SHA-256 補足。
- [preflight](preflight.json)、[DeepSpeed 失敗](deepspeed-before.json)／[單變數通過](deepspeed-after.json)、[BF16 dtype 量測及失敗回歸](bf16-before.json)。
- [驗證 driver](driver.py)、[最小 MHA 案例](minimal_mha.py)、[實際設定](config.yaml)。這些是固定實驗輸入副本；driver 使用原公開 CLI、保存四份報告、驗證跨程序輸出，並執行 pytest 與 NaN／Inf 注入。

最終整合版遠端快照：`/share/home/r15543056/MDgen/codex-runs/preflight-20261023T040605Z-4ebee2a5`。首次修正驗證的固定快照及原始輸出：`/share/home/r15543056/MDgen/codex-runs/preflight-20261023T035755Z-95f050bf`；code 唯讀，results/logs 分開。重跑時將 driver 與 minimal_mha 放回 manifest 的 `.scratch/l40s-validation/` 相對位置，以 `config.yaml` 的入口與明列檔案建立**新**快照再提交；保留舊 run 的結果。設定檔及兩份輸入副本本身不會提交 Job。

## 結論範圍

Ticket 2 的小尺寸 L40S 執行、報告、梯度比較、非有限值偵測與重現性驗證通過。這不是 Training region × 250 幀的容量驗收、7,500 殘基生成、正式資料／checkpoint、1,000 步訓練或 BF16 採用決策；沒有通過 Ticket 3／4 或模型品質驗證的宣稱。
