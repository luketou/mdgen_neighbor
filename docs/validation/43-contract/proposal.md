# Issue #43 Ticket 2：比較契約提案

狀態：使用者已於本次對話回覆「同意」，確認 forward 門檻、候選矩陣、測試 seam 與先導資源上限。容量先導尚未完成，最終矩陣與訓練比較規則尚未凍結。

來源：[Issue #43](https://github.com/luketou/mdgen_neighbor/issues/43)、其兩則留言、本地 `codex-handoff-issue43.md`，以及主 repo 的 `docs/validation/43-49427/README.md`、Halo／training-step probe 文件與程式。

## 一、建議判定規則

對每個受 supervision 的 loss-core 殘基 i，取未條件化幀與有效通道上的完整蛋白 local 輸出 y、region 輸出 z：

- `N_i`：有效分量數；不能把被 mask 的零值計入分母。
- 絕對 L2：`a_i = ||z_i - y_i||₂`。
- 參考範數：`r_i = ||y_i||₂`。
- 輔助尺度：`a_i / sqrt(N_i)`（輸出分量 RMS 差異）；另保存原始 `a_i`、`r_i` 與 `N_i`。
- 相對誤差：一般樣本為 `e_i = a_i / r_i`。

真正截斷的 case 建議採雙門檻：

| 指標 | p95 上限 | max 上限 |
|---|---:|---:|
| 相對誤差 `e_i` | 0.01 | 0.05 |
| 絕對 L2 的幀數尺度 `a_i / sqrt(N_i)` | 0.01 | 0.05 |

等價地，逐樣本的絕對 L2 上限尺度是 `0.01 * sqrt(N_i)` 與 `0.05 * sqrt(N_i)`，再對標準化後的樣本計算 p95／max；不是拿不同 `N_i` 的原始 L2 混合設一個固定數字。所有原始 L2 仍報 median、p95、max。p95 使用 NumPy `method="linear"`。

理由：將典型尾端的相對擾動預算設為 1%，並以 5% 限制單一殘基的極端誤差；絕對條件防止大參考輸出掩蓋大差值。除以有效分量數的平方根，避免短幀與 250 幀僅因維度不同而改變絕對容差。這些是事前提出的工程容差，沒有獲驗證的物理或下游品質校準；輸出含不同通道，不能解讀為 Å 容差。既有 ADR 記錄的一圈 halo 邊界誤差可能超過此提案，因此允許不通過，不據新先導結果放寬。

完整涵蓋 case 作工具一致性對照：每個殘基須滿足 `a_i <= 1e-5 * sqrt(N_i) + 1e-5 * r_i`；此為 FP32 工程容差提案，不宣稱由 machine epsilon 保證。完整涵蓋與截斷各自呈現，不混入同一通過率。

零分母與失敗政策：

- `N_i == 0`：列為無 supervision，獨立計數；不進一般誤差分布。
- `r_i == 0`：`zero_reference`；`0 < r_i / sqrt(N_i) <= 1e-6`：`near_zero_reference`。兩者的相對誤差均記 `null`，分別報分子、絕對分布及樣本數，不計成 NaN／Inf。
- 上述零／近零樣本仍須通過絕對條件；若某一分層沒有一般參考樣本，相對條件標為「無法評估」，不能宣告整體雙門檻通過。
- 任何原始輸入、輸出或差值的 NaN／Inf 均獨立計數，該 case 不通過；不能僅刪除後用 finite 分布判定。
- 不可行、OOM、缺 GPU、超時與程式失敗各自保留。缺乏必要結果的分層標示未完成。
- 每個非空分層均須符合門檻；不以跨蛋白 pooled 統計蓋過不通過的蛋白或邊界。

Ticket 5 的 loss／gradient／optimizer 數值門檻另需在正式執行前提出並確認；此文件的 forward 門檻不適用於梯度，也不能單憑本提案凍結整份 Ticket 2 契約。

## 二、先導與正式矩陣提案

固定四蛋白 `6sms_A`、`6lrd_A`、`6l3r_E`、`6sup_A`，R1–R3，各自使用 frame 0 建立的分區與起始視窗，400 ps 間隔，不再降採樣。checkpoint SHA-256：`bc90c66e7539f52c9d0514aeddd96c552061150a1eef3b10c120bc3aaff08015`。

| 項目 | 容量先導 | 待凍結的正式 forward 候選 |
|---|---|---|
| 幀數 | 4，再 250；分開報告 | 250 為目標，4 幀僅為獨立診斷層 |
| batch | 1 | 1 |
| t | 0.5 | 0.25、0.5、0.75 |
| 雜訊 seed | 0 | 0、1、2 |
| 空間預算 | 896；完整蛋白另作 reference | 優先 896，依容量規則確認後凍結 |
| halo | 1、2；每種均盤點全部 seed group | 1、2 與各自可行率、成對共同子集 |
| region 數 | 每蛋白／replica 最多 4 個固定候選 | 每蛋白／replica 最多 8 個固定候選 |

先在 CPU Compute Job 枚舉每蛋白／replica 的所有 group，分別列出 1／2 圈的最小種子 region 殘基數，報 min、median、p95、max、超預算 group ID 與原因。盤點是幾何與容量分析，不讀模型誤差。

候選種子 group 選法：對 group ID 以 CPU `torch.randperm`、generator seed 43002 固定排序，先導取前 4 個、正式候選取前 8 個（不足就全取）。不可行不補抽，重複 loss-group set 不補抽；保存所有候選到實際 region 的對應。這與目前 CLI 的 `--seed + r` 隨機抽種子不同，需補正式工具的顯式 group 選取能力。

共同子集：對相同 seed group 用 2 圈預算選 loss groups，再分別加 1／2 圈 halo。2 圈種子放不下時，另報 1 圈自身可行結果，不能將它混入成對 halo 比較。完整涵蓋定義為實際 region 殘基等於完整蛋白殘基；截斷定義為小於，均記錄實際殘基與 loss-group 集合。

若上述固定候選沒有真正截斷的 region，或關鍵的 2 圈 case 無法容納，先報缺口並提出容量決策；不根據誤差更換 seed、蛋白或預算。896 不可行不自動升為 1024／更大，也不將小蛋白完整涵蓋冒充長蛋白驗證。

聚合鍵：蛋白、replica、幀數、t、雜訊 seed、halo、邊界圈數、完整涵蓋／截斷與共同子集狀態。保存逐殘基原始誤差，再提供各雜訊 seed 與跨 seed 的描述統計。重複 region 按 loss-group set 去重，同一殘基在不同 region 的觀測保留並標 region ID；這些觀測不當成獨立蛋白。

每層報 median、p95、max、有效樣本數、zero／near-zero、nonfinite；另以候選 seed group 為分母報 attempted、可行、跳過、重複、去重後 region 數與覆蓋率，並報去重後 loss 殘基聯集占全蛋白的比例。`None` 邊界圈數獨立保留。

## 三、固定執行設定與待補能力

- FP32 reference、FP32 權重、matmul `highest`、TF32 關閉、dropout 關閉；完整蛋白與 region 使用同一份權重、輸入、雜訊、t、條件 mask、loss mask 與正規化。
- local cell width 16 Å、contact radius 16 Å、core capacity 128、ghost capacity 128；cell summaries 關閉、FFN chunk size 65536。實際生效狀態必須寫入 JSON。
- forward 為 eval／no-grad、gradient checkpointing 關閉；training-step 容量探針啟用 gradient checkpointing。不得混入 BF16 比較。
- 每個 case 記錄原始與 padded batch shape、actual region 大小、checkpointing／chunking 設定。
- 同一輸入暖機 1 次、量測 3 次；CUDA 同步時間與 peak allocated／reserved 分別保存。失敗時保留階段、退出碼與 log，不宣稱效能收益。
- 完整蛋白與 region 的訓練容量先導，只能稱各自流程／容量結果；在同 supervision 成對能力補齊前，不比較其 loss、gradient 或速度收益。

目前能力缺口（尚未修改）：

1. Halo CLI 缺顯式小幀數覆寫、上述完整統計、逐樣本零分母處理、全部 group 可行盤點與固定 group 清單；需補實際 precision 狀態與資源量測。
2. `job/submit_training_region_probe.sh` 固定合成資料；需讓正式入口支援真實資料容量先導。
3. training-step probe 現有比較是 precision 比較；checkpoint 權重載入與同 supervision 的 full／region 成對能力須確認或補齊，不能以隨機初始化模型代替指定 checkpoint 的容量結果。

程式修改依 TDD 執行。建議測試 seam 為正式 Halo CLI／JSON（驗證篩選、去重、零分母、分層與失敗揭露）及正式 training-step probe／JSON（驗證真實資料、權重與 batch 設定）；合成 CPU 測試不驗證 L40S 容量或 MD 品質。待確認這兩個 seam 後才新增測試。

## 四、建議先導資源上限

以下先導額度已獲使用者確認；不含正式 Ticket 3–5 的 GPU 執行。

- CPU：最多 2 個 Job，每個 2 CPU、16 GiB、30 分鐘，供 group 統計與工具測試。
- GPU：最多 4 個 Job，逐一執行，不併行；每個 1 張 L40S、2 CPU、32 GiB 主記憶體、30 分鐘，合計最多 2 GPU 小時的配置上限；preflight 失敗及重送亦計入上限。
- 先完成小幀 case，再嘗試 250 幀；完整蛋白與 region、forward 與 backward 分開記錄失敗。資源內未跑完的矩陣列為未完成，不自動加額。
- 只使用 `scripts/hpc_run.py` 與正式入口。新快照先通過適用檢查；提交前保存參數、來源 commit／manifest；不覆寫既有資料、結果或 checkpoint。

## 五、確認與後續凍結

本次已確認：forward 工程門檻及零分母政策、候選矩陣與容量決策規則、兩個 TDD seam，以及 CPU／GPU 先導資源上限。後續只在這個額度內提交。

確認後先補工具並審查／提交，再依核准上限量測容量。先導結束後，連同 Ticket 5 的待確認判定規則，將最終可行矩陣、全部不可行 case、完整 hash／設定與確認紀錄寫入 repo 的 `docs/validation/43-contract/`；Ticket 2 此前維持未完成。
