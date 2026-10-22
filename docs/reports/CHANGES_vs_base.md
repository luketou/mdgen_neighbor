# MDGen-2026 相對 base 的最終改動

本文件整理目前版本相對上游 commit `81482a4` 的最終差異。內容只採用 `MDGEN2026_REPORT-r1.md`、`MDGEN2026_REPORT-r2.md` 與 `MDGEN2026_REPORT-r2.1.md` 已記載的改動及 `block_name`，不另外命名。r1 曾加入、但已在 r2／r2.1 撤銷的 block 不列入最終變更。

本文件使用三種標記整理改動與驗證：

| 標記 | 定義 | 用途 |
|---|---|---|
| Bx | 依用途整理出的改動主題，例如 SDPA、Position ID、precision、資料前處理或 runtime 相容性。 | 協助讀者從功能面理解改動；Bx 不是 code 裡的註解。一個 `block_name` 只歸入一個 Bx。 |
| `block_name` | r1、r2、r2.1 report 已定義，並對應到 code 中實際註解錨點的名稱。 | 用來定位實際修改的程式區塊；本文件不自行創造 `block_name`。 |
| Cx | report 中執行 code、smoke test、regression 或 benchmark 的驗證案例編號。 | 只在 Validation 中說明執行結果並連結 log／JSON／txt 等證據；Cx 不是 `block_name`。 |

閱讀方式：(a) 先看 B1～B8 總覽，掌握改動目的；(b) 再看新增檔案，知道 repository 多了哪些程式、文件與環境設定；(c) 最後依 B1～B8 查閱各 `block_name` 的最終行為。同一個 `block_name` 只歸入一個 Bx，同一個 `file_path`／`block_name` 組合只列一次。

## B1～B8 改動總覽

| Bx | 主題 | 為什麼要改 | 最終結果 |
|---|---|---|---|
| B1 | SDPA 與 attention 語意 | 導入 PyTorch SDPA，同時保留上游 manual attention 作為比較路徑。 | 正式入口預設使用 SDPA；scaling、mask、bias K/V、dropout、layout 與 time-axis 行為均明確對齊。 |
| B2 | Position ID 與 RoPE 介面 | 預留 position ID 介面，但避免尚未完成的 true-gap 語意被靜默使用。 | 正常流程固定傳 `None`；任何非 `None` 值在 MHA 入口明確拒絕；CSV 產生工具保留供未來使用。 |
| B3 | FP32 與 NPY dtype | 分開模型運算精度、matmul policy 與資料儲存 dtype。 | 訓練精度改為具名組態，預設 `fp32-reference`（FP32、`highest` matmul、TF32 關閉），可選 `bf16-train`（#9 Ticket 1 取代 r2 的上游 `medium` matmul）；推論模型轉 FP32；NPY 預設恢復上游 float16，也可選 float32。 |
| B4 | Seed 與 deterministic | 一般執行不應被 benchmark 的可重現設定綁住。 | seed 與 deterministic 預設關閉，需要可重現測試時再由 CLI 明確啟用。 |
| B5 | Runtime、checkpoint 與入口分工 | 支援新版 PyTorch／舊 checkpoint，並避免量測參數污染正式訓練入口與 checkpoint metadata。 | 正式入口、runtime 入口與 checkpoint runtime 參數分離，可信任舊 checkpoint 以 `weights_only=False` 載入。 |
| B6 | 資料下載與前處理 | 提供 ATLAS 下載工具，並修正資料 window 邊界與路徑參數不一致。 | ATLAS 資料可由 repository 工具下載；最後合法 window 可被抽到，短軌跡處理正確，`--sim_dir` 與 `--atlas_dir` 相容。 |
| B7 | 驗證、效能與 regression | 證明 manual／SDPA 數值等價、實際 backend、memory、速度與長度擴展行為。 | 提供獨立 regression、runtime training 與 L-scaling 工具，並保存各輪驗證輸出。 |
| B8 | 環境、repository 與來源追蹤 | 讓環境可重建、大型產物不進 Git，第三方程式可追溯。 | 提供 PyEMMA 環境鎖定檔、artifact ignore 規則、第三方來源授權與三份回報文件。 |

## 新增檔案

本表只介紹新增的程式、repository 文件及實際會使用的環境設定，不列出 `my_new/` 內保存的 log、JSON、txt、CSV 等 validation artifacts。新增檔案以「整個檔案」為單位說明，因此不替沒有既有 `block_name` 的檔案創造名稱；每一列仍指定其對應 Bx。

| 新增檔案 | 關聯 Bx | 內容與新增原因 |
|---|---|---|
| `.gitignore` | B8 | 排除 checkpoint、workdir、wandb、暫存檔，以及 `*.npy`、`*.npz`、`*.pdb`、`*.pickle` 等大型產物，避免驗證輸出誤入 Git。 |
| `MDGEN2026_REPORT-r1.md` | B8 | 記錄第一輪實作、block 定義、執行方法與 T4／T5 驗證結果。 |
| `MDGEN2026_REPORT-r2.md` | B8 | 逐項回覆 r1 review 的 M1～M12，並記錄各項撤銷、修正與驗證結果。 |
| `MDGEN2026_REPORT-r2.1.md` | B8 | 記錄 r2.1 的最終處置、補充驗證與結案狀態。 |
| `train-runtime.py` | B5、B7 | 將 backend profiler、peak memory 與 sec/step 量測從正式 `train.py` 分離，供 training benchmark 使用。 |
| `benchmark_l_scaling.py` | B4、B5、B7 | 逐一量測不同 sequence length 的單次 model forward、peak memory、OOM 與實際 attention backend。 |
| `test_sdpa.py` | B1、B2、B7 | 比較 manual／SDPA 的 output、input gradient 與 parameter gradient，並涵蓋 no-RoPE padding、position ID 拒絕與整列 padding。 |
| `mdgen/model/mha_legacy.py` | B7 | 固定保存上游 manual MHA，作為不隨目前 manual 分支改動的 regression baseline；正式模型不匯入。 |
| `scripts/prep-protein-csv.py` | B2 | 從 PDB 建立 `seqres` 與 `position_ids` CSV；position IDs 只供未來使用，目前模型不讀取。 |
| `my_new/download_atlas.sh` | B6 | 下載 ATLAS 前處理所需的 simulation data；整個檔案是新增工具，因此不另外創造 `block_name`。 |
| `scripts/analyze_ensemble.py` | B8 | 從 AlphaFlow 引入的 ensemble 分析工具，保留固定來源與 MIT 授權資訊。 |
| `scripts/print.py` | B8 | 從 AlphaFlow 引入的分析結果彙整工具，保留固定來源與 MIT 授權資訊。 |
| `my_new/environment-mdgen2026-cu126.yml`、`my_new/environment-mdgen2026-cu130.yml` | B8 | 保存 r1 cu126／cu130 歷史環境，檔頭標示為歷史用途。 |
| `my_new/environment-mdgen2026-cu126-pyemma.yml`、`my_new/environment-mdgen2026-cu130-pyemma.yml` | B8 | 加入 PyEMMA 與實際解析依賴的 r2 環境鎖定檔；cu126 版本是正式環境，cu130 版本供硬體追蹤。 |

## B1 — SDPA 與 attention 語意

| block_name | file_path | 最終改動 | 原因 |
|---|---|---|---|
| `sdpa-route` | `mdgen/parsing.py`、`train.py`、`sim_inference.py`、`tps_inference.py`、`design_inference.py`、`upsampling_inference.py`、`train-runtime.py`、`benchmark_l_scaling.py` | 正式入口未帶 flag 時使用 SDPA；只有 `--manual_attention` 才使用上游 manual attention。 | 讓加速路徑成為預設，同時保留可比對的 manual 路徑。 |
| `sdpa-route` | `mdgen/wrapper.py`、`mdgen/model/latent_model.py`、`mdgen/model/mha.py` | wrapper、model、layer 與 MHA 的 `use_sdpa` 預設為 `True`，並向下傳遞路徑選擇；MHA 記錄 fallback。 | 即使直接建立底層類別，也要與正式入口的預設一致。 |
| `sdpa-scaling` | `mdgen/model/mha.py` | 保留原本 `q *= self.scaling`，呼叫 SDPA 時指定 `scale=1.0`。 | 避免 query scaling 被套用兩次。 |
| `sdpa-mask` | `mdgen/model/mha.py` | 將 fairseq padding mask 轉為 SDPA 的保留 mask；沒有實際遮蔽時傳 `None`。 | 對齊 manual 與 SDPA 的 mask 語意並避免不必要 mask。 |
| `sdpa-bias-kv` | `mdgen/model/mha.py` | 保留 `add_bias_kv=True` 的 K／V token append，並同步延長 mask。 | MDGen 實際使用 bias K/V，不能在改用 SDPA 時遺失。 |
| `sdpa-dropout` | `mdgen/model/mha.py` | training 使用原 dropout，evaluation 傳入 `0.0`。 | SDPA 的 dropout 不會自動依 `eval()` 關閉。 |
| `sdpa-layout-output` | `mdgen/model/mha.py` | SDPA 前轉成 `B,H,L,D`，完成後還原 layout，再通過既有 `out_proj`。 | 維持原 MHA 的輸入輸出形狀與 projection 行為。 |
| `sdpa-time-axis-mask` | `mdgen/model/latent_model.py` | time-axis attention 沿用同一套 mask 與 SDPA 轉換流程。 | residue 與 time 兩軸都必須保持遮蔽語意一致。 |
| `no-rope-padding-mask` | `mdgen/model/latent_model.py` | caller 將有效位置 mask 轉為 boolean `key_padding_mask`：有效 residue 為 `False`，padding 為 `True`。 | 修正 MDGen 2024 在 `--no_rope` 下把 float mask 當 additive mask、未真正遮蔽 padding key 的行為。 |

## B2 — Position ID 與 RoPE 介面

| block_name | file_path | 最終改動 | 原因 |
|---|---|---|---|
| `position-id-rope-interface` | `mdgen/model/mha.py` | 保留 `position_ids=None` 介面與上游連續 RoPE；任何非 `None` 值統一在 `MultiheadAttention.forward` 入口拋出 `NotImplementedError`。 | true-gap／multi-chain 語意尚未實作，不能靜默忽略使用者輸入。 |
| `position-id-model-routing` | `mdgen/model/latent_model.py` | 模型介面保留 `position_ids=None`，可向 residue-axis attention 傳遞；正常訓練與推論不提供非 `None` 值。 | 保留未來擴充介面，但不改變目前模型行為。 |
| `position-id-residue-layout` | `mdgen/model/latent_model.py` | 保留將 `B,L` position ID 展開成 residue attention 使用的 `B×T,L` layout；time-axis 不使用。 | 明確界定 position ID 只屬於 residue 軸。 |
| `position-id-csv` | `scripts/prep-protein-csv.py` | 從 PDB 建立 `seqres` 與保留 chain gap 的 `position_ids` CSV；檔頭註明 r2.1 模型不讀取此欄位。 | 保存未來 position-aware fork 所需的資料準備工具，不誤導目前已支援。 |

## B3 — FP32 與 NPY dtype

| block_name | file_path | 最終改動 | 原因 |
|---|---|---|---|
| `fp32-training-precision` | `mdgen/parsing.py` | r2.1 時 `--precision` 只允許 `32-true`；#9 Ticket 1 已移除此錨點，改為具名組態 `fp32-reference`（預設）／`bf16-train`，並接受 `32-true`／`bf16-mixed` 別名。 | 精度須明確選擇並記錄，`fp32-reference` 作為 Precision parity 的參考。 |
| `fp32-matmul-precision` | `train.py`、`train-runtime.py` | r2.1 時使用上游的 `torch.set_float32_matmul_precision("medium")`；#9 Ticket 1 已移除此錨點，改由 `mdgen/precision.py` 的 `configure_precision()` 依組態設定 matmul 與 TF32，並在啟動時記錄實際生效狀態。 | 全域 `medium` 讓「FP32」訓練的 matmul 內部允許 BF16，不是乾淨的參考。 |
| `fp32-inference-model` | `sim_inference.py` | checkpoint 載入後使用 `model.eval().float()`。 | 明確保證推論模型參數為 FP32。 |
| `npy-dtype-option` | `scripts/prep_sims.py` | 新增 `--dtype {float16,float32}`，預設 `float16`，依選項決定 NPY 儲存 dtype。 | 預設對齊 ATLAS／`atlas.ckpt` 的上游量化方式，較大座標範圍仍可選 float32。 |

## B4 — Seed 與 deterministic

| block_name | file_path | 最終改動 | 原因 |
|---|---|---|---|
| `seed-deterministic-args` | `mdgen/parsing.py`、`sim_inference.py`、`benchmark_l_scaling.py` | training／inference／L-scaling 的 seed 預設為 `None`，deterministic 預設關閉，仍保留顯式 CLI 控制。 | 一般執行不應自動承擔 deterministic 的效能與環境限制。 |
| `seed-initialization` | `train.py`、`train-runtime.py`、`sim_inference.py`、`benchmark_l_scaling.py` | 只有明確提供 seed 時才初始化 RNG。 | 區分一般執行與需要重現的 benchmark／測試。 |
| `deterministic-execution` | `train.py`、`train-runtime.py`、`sim_inference.py`、`benchmark_l_scaling.py` | training 由 Lightning Trainer 控制，inference／L-scaling 依 CLI 設定 deterministic algorithms。 | 讓各入口使用一致、可顯式選擇的執行政策。 |
| `deterministic-benchmark-guard` | `mdgen/parsing.py`、`train.py`、`train-runtime.py`、`sim_inference.py` | 禁止 deterministic algorithms 與 cuDNN benchmark 同時啟用。 | 避免互相衝突的設定造成不明確行為。 |

## B5 — Runtime、checkpoint 與入口分工

| block_name | file_path | 最終改動 | 原因 |
|---|---|---|---|
| `runtime-compatibility` | `mdgen/model/primitives.py`、`mdgen/tensor_utils.py` | legacy FlashAttention 改為 optional import，並將新版 PyTorch 不接受的 list indexing 改為 tuple indexing。 | 讓沒有 FlashAttention 的 PyTorch 2.x 環境可匯入並執行。 |
| `runtime-compatibility` | `train.py`、`train-runtime.py`、`sim_inference.py`、`tps_inference.py`、`design_inference.py`、`upsampling_inference.py`、`benchmark_l_scaling.py` | 載入可信任的舊 Lightning checkpoint 時明確使用 `weights_only=False`。 | PyTorch 2.6 之後的預設會拒絕含 `argparse.Namespace` 的舊 checkpoint。 |
| `checkpoint-runtime-args` | `mdgen/wrapper.py` | `save_hyperparameters` 只保存原有 `args`，不把 `use_sdpa` 等 runtime 選項寫成 checkpoint 建構需求。 | 保持舊 checkpoint 的建構介面與可載入性。 |
| `runtime-args-separation` | `mdgen/parsing.py`、`train.py`、`train-runtime.py` | 正式 `train.py` 不註冊量測 instrumentation；backend profiler、peak memory 與 execution time 只由 `train-runtime.py` 使用。 | 避免 benchmark 參數污染正式 training 入口。 |
| `model-output-dir` | `mdgen/parsing.py` | 新增 `--model_dir` 指定 training 輸出根目錄；未提供時維持 `workdir/<run_name>`。 | 允許不同儲存環境調整輸出位置，同時保持上游預設。 |

## B6 — 資料下載與前處理

| block_name | file_path | 最終改動 | 原因 |
|---|---|---|---|
| `data-preprocess-fixes` | `mdgen/dataset.py` | 修正 frames 不足時的處理錯誤，並讓最後一個合法時間窗可被抽到。 | 避免短軌跡出錯與 time-window sampling 少掉合法起點。 |
| `data-preprocess-fixes` | `scripts/prep_sims.py` | 統一程式使用 `args.atlas_dir`；`--atlas_dir` 與舊 `--sim_dir` 解析到同一欄位。 | 修正欄位名稱不一致，同時保留 README 舊指令相容性。 |

## B7 — 驗證、效能與 regression

| block_name | file_path | 最終改動 | 原因 |
|---|---|---|---|
| `sdpa-diagnostics` | `mdgen/model/mha.py`、`train-runtime.py`、`benchmark_l_scaling.py` | 記錄各 MHA 最後實際使用的 manual／SDPA 路徑、fallback reason，並可輸出實際 SDPA operator／backend。 | 驗證「設定使用 SDPA」確實對應到 runtime 呼叫與 backend。 |
| `peak_memory` | `mdgen/parsing.py`、`train-runtime.py` | 成功或 OOM 時輸出 CUDA peak allocated／reserved、完成 step 與執行設定。 | 提供 A2／A3 可比較的 memory 證據。 |
| `execution_time` | `mdgen/parsing.py`、`train-runtime.py` | 使用 CUDA Event 量測完整 training step，排除前五個 warm-up step後輸出統計。 | 提供不含 warm-up 的 sec/step 比較。 |
| `a4-l-scaling` | `benchmark_l_scaling.py` | 逐 L 記錄 peak memory、OOM、matmul precision、deterministic 狀態與各 MHA backend。 | 評估 manual／SDPA 隨序列長度增加的擴展行為。 |
| `sdpa-equivalence-test` | `test_sdpa.py` | 比較 manual／SDPA forward、input gradient 與 parameter gradient；固定 `highest` 並關閉 TF32，也測試 no-RoPE padding invariance 與非 `None position_ids` 拒絕。 | 在排除低精度差異後驗證兩條 attention 路徑的數值等價與介面行為。 |
| `sdpa-upstream-reference` | `mdgen/model/mha_legacy.py`、`test_sdpa.py` | RoPE residue／time 測試以固定的上游 legacy manual 實作對照目前 SDPA。 | 避免目前 manual 分支與 SDPA 同時改錯卻彼此相等。 |
| `sdpa-time-full-row-padding` | `test_sdpa.py` | 新增 time-axis 整列 padding case，檢查 output／gradient 誤差與 finite 狀態。 | 涵蓋真實 time keys 全被遮蔽、只剩 bias K/V token 的邊界情況。 |

## B8 — 環境、repository 與來源追蹤

| block_name | file_path | 最終改動 | 原因 |
|---|---|---|---|
| `runtime-environment` | `my_new/environment-mdgen2026-cu126-pyemma.yml`、`my_new/environment-mdgen2026-cu130-pyemma.yml` | 鎖定 PyEMMA 2.5.12、deeptime 0.4.5 與相關依賴；環境名稱與 r1 區隔，cu126 為正式檔。 | TPS／design 入口會經 `mdgen.analysis` 匯入 PyEMMA，環境必須可直接重建。 |
| `repository-artifacts` | `.gitignore` | 排除 checkpoint、workdir、wandb、暫存檔與大型 array／structure／pickle 產物。 | repository 只保存程式、環境與小型可審查證據。 |
| `third-party-provenance` | `scripts/analyze_ensemble.py`、`scripts/print.py` | 記錄 AlphaFlow 固定來源 commit、原始檔名、作者 copyright 與 MIT License；分析邏輯不變。 | 讓第三方程式的來源、版本與授權可追溯。 |

## 刪除的檔案

無。r2／r2.1 撤銷的是部分 code block 與大型產物的 Git 追蹤，沒有相對 base 刪除完整程式檔。

## 與 MDGen 2024 的刻意行為差異

MDGen 2024 在 `--no_rope` 且 batch 含 padding 時，manual attention 會把 float `1 - mask` 傳給 PyTorch `F.multi_head_attention_forward`；該值被解讀成加到 attention score 的 additive mask，而不是 padding 遮蔽，因此 padding key 並未真正被排除。MDGen-2026 以 `no-rope-padding-mask` 在 attention caller 建立 boolean `key_padding_mask`，使有效 residue 為 `False`、padding 為 `True`，並讓 RoPE／no-RoPE、manual／SDPA 共用相同遮蔽語意。

此差異只影響 `--no_rope` 且輸入含 padding 的訓練或推論。預設啟用 RoPE 的 ATLAS 設定與 `atlas.ckpt` 不受影響；若 checkpoint 是以 MDGen 2024 的 `--no_rope` 行為訓練，含 padding 輸入的結果可能與 MDGen-2026 不同。

## 資料面改動

| 項目 | 最終行為 | 影響與原因 |
|---|---|---|
| ATLAS time window | `mdgen/dataset.py` 修正 sampling 上界，使最後一個合法 window 可被抽到。實測 ATLAS NPY 有 251 frames：上游的條件 frame 恆為 frame 0，修正後一半樣本落在 frame 1，兩者只差 400 ps。 | 修正原本少一個合法起點的 off-by-one；資料抽樣分布有小幅改變，但不改變模型架構。 |
| NPY dtype | `scripts/prep_sims.py --dtype` 支援 `float16`／`float32`，預設為上游使用的 `float16`。dataset 讀入後仍轉成 float32，模型運算 dtype 不變。 | 預設維持 ATLAS 與 `atlas.ckpt` 的資料量化條件；Piezo1、膜蛋白或 ligand 等較大座標範圍可明確選擇 float32。 |

# Validation

本章依 B1～B8 整理最終版本的驗證結果。`Code block` 是 r1、r2 或 r2.1 report 中的 Cx 驗證案例編號，不是 `block_name`；`—` 表示該項只有 code review，沒有另外保存獨立 runtime case。完整執行指令與過程仍以三份 report 為準。

| Bx | 需要驗證什麼 | Code block | 最終結果 | 驗證檔案 |
|---|---|---|---|---|
| B1 | Manual 與 SDPA 的 output、input gradient、parameter gradient 是否等價 | `C32` | residue／time、padding／no-padding cases 全部通過既有誤差門檻。 | [C32_M11-regression-cu126.log](../validation/mdgen2026/r2/C32_M11-regression-cu126.log)、[summary.json](../validation/mdgen2026/r2/C32_M11-regression-cu126/summary.json) |
| B1 | `--no_rope` 且含 padding 時是否真正遮蔽 padding key | `C32` | Manual／SDPA 的 padding invariance error 均為 0。 | [C32_M11-regression-cu126.log](../validation/mdgen2026/r2/C32_M11-regression-cu126.log)、[summary.json](../validation/mdgen2026/r2/C32_M11-regression-cu126/summary.json) |
| B1 | 正式 training 未指定 attention flag 時是否實際使用 SDPA | `C39` | 實際呼叫 SDPA，並記錄到 `EFFICIENT_ATTENTION` backend。 | [C39_M8-train-default-sdpa-backend.log](../validation/mdgen2026/r2/C39_M8-train-default-sdpa-backend.log)、[C39_M8-train-default-sdpa-backend.txt](../validation/mdgen2026/r2/C39_M8-train-default-sdpa-backend.txt) |
| B1 | 底層 MHA、IPA 與 latent layer 未指定 `use_sdpa` 時是否預設使用 SDPA | `C42` | 各底層類別的預設 SDPA smoke test 通過。 | [C42_R5-default-sdpa.log](../validation/mdgen2026/r2.1/C42_R5-default-sdpa.log) |
| B1 | Manual 與預設 SDPA 的正式入口是否均可執行 | `C24`～`C26` | 預設 SDPA training 完成 10 steps；Manual 與預設 SDPA inference 均成功產生 PDB。 | [C24_M4-C16-run.log](../validation/mdgen2026/r2/C24_M4-C16-run.log)、[C25_M4-C17-run.log](../validation/mdgen2026/r2/C25_M4-C17-run.log)、[C26_M4-C18-run.log](../validation/mdgen2026/r2/C26_M4-C18-run.log) |
| B2 | 非 `None position_ids` 是否在 MHA 入口統一拒絕 | `C41` | RoPE／no-RoPE、Manual／SDPA 均拋出 `NotImplementedError`，其餘 regression cases 亦全部通過。 | [C41_R4-position-ids.log](../validation/mdgen2026/r2.1/C41_R4-position-ids.log)、[summary.json](../validation/mdgen2026/r2.1/C41_R4-position-ids/summary.json) |
| B2 | CSV 產生工具是否保留 `position_ids` 欄位，且目前模型不讀取 | — | 已完成程式與檔頭說明的 code review；沒有保存獨立 runtime case。 | [MDGEN2026_REPORT-r2.md](MDGEN2026_REPORT-r2.md) |
| B3 | 恢復 `medium` matmul 後，SDPA、gradient checkpointing off 的 training 是否可執行 | `C21` | 100-step runtime 完成，mean 為 0.4463 sec/step。 | [C21_M2-C2-run.log](../validation/mdgen2026/r2/C21_M2-C2-run.log)、[C21_M2-C2-time.json](../validation/mdgen2026/r2/C21_M2-C2-time.json) |
| B3 | 恢復 `medium` matmul 後，Manual、gradient checkpointing on 的 training 是否可執行 | `C22` | 100-step runtime 完成，mean 為 0.7629 sec/step。 | [C22_M2-C5-run.log](../validation/mdgen2026/r2/C22_M2-C5-run.log)、[C22_M2-C5-time.json](../validation/mdgen2026/r2/C22_M2-C5-time.json) |
| B3 | 恢復 `medium` matmul 後，SDPA、gradient checkpointing on 的 training 是否可執行 | `C23` | 100-step runtime 完成，mean 為 0.5843 sec/step。 | [C23_M2-C6-run.log](../validation/mdgen2026/r2/C23_M2-C6-run.log)、[C23_M2-C6-time.json](../validation/mdgen2026/r2/C23_M2-C6-time.json) |
| B3 | NPY 的 float16 預設與 float32 選項是否正確接線 | — | 已完成兩個 dtype 分支的 code review；沒有保存獨立 NPY 產生 runtime case。 | [MDGEN2026_REPORT-r2.md](MDGEN2026_REPORT-r2.md) |
| B4 | 明確啟用 deterministic 時，Manual／SDPA L-scaling 是否均可執行 | `C37`、`C38` | 兩條路徑均完成，JSON 亦保存 deterministic 與 matmul precision 設定。 | [C37_E2-cu130-manual-l-scaling.log](../validation/mdgen2026/r2/C37_E2-cu130-manual-l-scaling.log)、[C37_E2-cu130-manual-l-scaling.json](../validation/mdgen2026/r2/C37_E2-cu130-manual-l-scaling.json)、[C38_E2-cu130-sdpa-l-scaling.log](../validation/mdgen2026/r2/C38_E2-cu130-sdpa-l-scaling.log)、[C38_E2-cu130-sdpa-l-scaling.json](../validation/mdgen2026/r2/C38_E2-cu130-sdpa-l-scaling.json) |
| B4 | Runtime metadata 是否保存 deterministic algorithms 的實際狀態 | `C45` | 輸出 JSON 已包含 deterministic、matmul precision 與各 MHA backend。 | [C45_E2-runtime-metadata.log](../validation/mdgen2026/r2.1/C45_E2-runtime-metadata.log)、[C45_E2-runtime-metadata.json](../validation/mdgen2026/r2.1/C45_E2-runtime-metadata.json) |
| B5 | 正式 training 入口是否可解析新預設並載入既有 checkpoint | `C28` | checkpoint 載入成功並完成一個 validation batch。 | [C28_M8-train-entry.log](../validation/mdgen2026/r2/C28_M8-train-entry.log) |
| B5 | TPS、design、upsampling 入口是否可解析新預設並載入既有 checkpoint | `C29`～`C31` | 三個入口均成功載入 checkpoint；空 split smoke test 未執行 attention forward。 | [C29_M8-tps-load.log](../validation/mdgen2026/r2/C29_M8-tps-load.log)、[C30_M8-design-load.log](../validation/mdgen2026/r2/C30_M8-design-load.log)、[C31_M8-upsampling-load.log](../validation/mdgen2026/r2/C31_M8-upsampling-load.log) |
| B5 | 新版 PyTorch 是否可使用既有 checkpoint 執行 validation | `C43` | checkpoint 載入成功並完成 validation。 | [C43_E1-validation-checkpoint.log](../validation/mdgen2026/r2.1/C43_E1-validation-checkpoint.log) |
| B5 | 既有 checkpoint 是否可恢復 training | `C44` | 成功載入來源 checkpoint，並完成 one-step fit resume。 | [C44_E1-source-checkpoint.log](../validation/mdgen2026/r2.1/C44_E1-source-checkpoint.log)、[C44_E1-one-step-resume.log](../validation/mdgen2026/r2.1/C44_E1-one-step-resume.log) |
| B6 | 實際 ATLAS NPY 是否為 251 frames | `C27` | 3 個蛋白、各 3 個 replica，共 9 個 NPY 均為 251 frames。 | [C27_M7-frame-count.log](../validation/mdgen2026/r2/C27_M7-frame-count.log) |
| B6 | Time-window 上界修正是否涵蓋最後合法起點與短軌跡 | — | 已完成 sampling 邊界的 code review；沒有保存獨立 sampling distribution runtime case。 | [MDGEN2026_REPORT-r2.md](MDGEN2026_REPORT-r2.md) |
| B7 | Regression 是否使用固定上游 Manual baseline，並涵蓋整列 time padding | `C32` | 上游 baseline、full-row padding、position ID 與 no-RoPE cases 全部通過。 | [C32_M11-regression-cu126.log](../validation/mdgen2026/r2/C32_M11-regression-cu126.log)、[summary.json](../validation/mdgen2026/r2/C32_M11-regression-cu126/summary.json) |
| B7 | Training memory snapshot 是否能保存 peak allocated／reserved 與執行狀態 | `C36` | SDPA training memory snapshot 已成功輸出 JSON。 | [C36_section3-C2-memory.json](../validation/mdgen2026/r2/C36_section3-C2-memory.json) |
| B7 | Manual／SDPA L-scaling 是否保存 peak memory、OOM 與實際 backend | `C37`、`C38` | 兩組 L-scaling 完成；SDPA 組另保存實際 backend profiler 輸出。 | [C37_E2-cu130-manual-l-scaling.json](../validation/mdgen2026/r2/C37_E2-cu130-manual-l-scaling.json)、[C38_E2-cu130-sdpa-l-scaling.json](../validation/mdgen2026/r2/C38_E2-cu130-sdpa-l-scaling.json)、[C38_E2-cu130-sdpa-backend.txt](../validation/mdgen2026/r2/C38_E2-cu130-sdpa-backend.txt) |
| B7 | Runtime JSON 是否保存足以重現量測的執行設定 | `C45` | JSON 包含 matmul precision、deterministic 狀態與各 MHA 的 backend／fallback reason。 | [C45_E2-runtime-metadata.log](../validation/mdgen2026/r2.1/C45_E2-runtime-metadata.log)、[C45_E2-runtime-metadata.json](../validation/mdgen2026/r2.1/C45_E2-runtime-metadata.json) |
| B8 | cu126／cu130 PyEMMA 環境能否由 YAML 從零建立 | `C19` | 兩套環境均成功建立。 | [C19_M12-cu126-conda-create.log](../validation/mdgen2026/r2/C19_M12-cu126-conda-create.log)、[C19_M12-cu130-conda-create.log](../validation/mdgen2026/r2/C19_M12-cu130-conda-create.log) |
| B8 | 正式環境的 PyTorch、Lightning、PyEMMA、deeptime 與 `mdgen` 是否可匯入 | `C20` | cu126／cu130 均完成 import 與版本確認。 | [C20_M12-cu126-import-version.log](../validation/mdgen2026/r2/C20_M12-cu126-import-version.log)、[C20_M12-cu130-import-version.log](../validation/mdgen2026/r2/C20_M12-cu130-import-version.log) |
