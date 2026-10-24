# 訓練步探針（#9 Ticket 2，Seam 1）

對實際的 `NewMDGenWrapper` Lightning 訓練模組跑一次完整 training step
（loss、backward、gradient clipping、optimizer step），輸出 JSON 報告：

```bash
python -m mdgen.train_step_probe --output results/probe.json \
    --configs fp32-reference bf16-train --seed 0 --dataset_index 0 \
    -- <與 train.py 相同的參數>
```

`--` 之後的參數交給 `parse_train_args`，模型組態（層數、`--local_spatial`、
`--grad_checkpointing`、`--batch_size`、`--crop` 等）與正式訓練相同；其中的
`--precision` 會被 `--configs` 取代。資料來源二擇一：

- `--dataset_index I`（預設 0）：取 `--train_split` 第 I 起連續 `--batch_size` 筆。
- `--synthetic_residues L`：隨機漫步骨架（Cα 間距 3.8 Å、合法旋轉、單位 torsion 向量），
  免資料、只用於記憶體與流程檢查；`--train_split`／`--val_split`／`--data_dir` 仍須給值但不會讀取。

`--sim_ckpt PATH`（放在分隔符之前）從指定 checkpoint 初始化權重及模型架構／條件設定，保留 CLI 的資料、空間容量與執行設定；不恢復 optimizer。載入不相容時直接失敗，報告保存完整檔案 SHA-256 與初始參數 norm。`--time T` 固定 transport 時間；未指定則維持原本隨機時間。`batch.frame_starts` 與 `runs[].training_time` 記錄實際視窗與時間。

真實 ATLAS 容量探針可用 `--atlas --replicates R --overfit_frame` 固定 replica 與起始幀；完整蛋白模式的 `--crop` 必須明確等於蛋白長度。`--ffn_chunk_size` 可顯式固定 FFN chunking。完整蛋白與 Training region 各自的容量結果不代表相同 supervision 的 loss／gradient 比較。

`--seed_group G`（分隔符之前）明確指定 Training region 的候選 group；實際 `seed_group`、`loss_group_ids`、`halo_group_ids` 保存在 region 摘要。先導 driver 使用核准的候選順序，各 halo 設定內按 loss-group 集合去重，保留重複與不可行候選的對應，不補抽。

`--coverage_control` 執行 FP32 的同路徑重跑與完整涵蓋對照：需要 batch 1、local 模式、dropout 關閉，且 region 的全部真實殘基均受監督。先生成完整雜訊再映射到 padded region，兩側使用相同非條件幀、有效通道及 loss 分母；JSON 的 `alignment` 記錄實際 noise／mask hash 與分母。未給 `--time` 時固定為 0.5，避免不同 shape 消耗 RNG 後抽到不同時間。

控制結果記錄每個參數群的 gradient／實際更新 norm、RMS、差向量、cosine 與精確零分類，以及 optimizer 設定和 clipping 前後 norm。零向量的 cosine 為 `null`；此模式尚不套近零 cutoff 或接受門檻。`comparisons` 區分 `same_path_repeat` 與 `full_coverage`，供控制容差校準；原本 precision probe 的 cosine 規則保持獨立，不能作為截斷比較的科學門檻。

剩餘先導使用核准的八候選與長蛋白優先順序；三個基準蛋白 R1 加入完整涵蓋控制。`scripts/issue43_pilot.py --plan_only` 可在 CPU Compute Job 驗證檔案與候選並產出命令清單，不執行模型。計畫與實際執行由 manifest 的 `plan_only` 與 `driver_finished` 區分。

既有 launcher 的 `MDGEN_ISSUE43_FRAMES` 模式會呼叫有時間上限的真實資料先導；需要明確設定 `MDGEN_ISSUE43_DATA` 與 `MDGEN_ATLAS_CKPT`。每個 case 重建相同初始權重與 optimizer，第一個 FP32 組態作暖機，其餘為量測。量測數及核准資源見 [提案](43-contract/proposal.md)，未跑到的 case 亦保留在 manifest。

`MDGEN_ISSUE43_MODE` 模式把 launcher 參數直接交給 driver，供 [幀數掃描申請](43-contract/gpu-request-frames.md) 使用：

- `--frames` 接受任意正整數。`--memory gbs=N,ffn=N,expandable` 覆寫 `group_batch_size`、`ffn_chunk_size`，並對子程序設定 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；未指定時沿用核准的 4／65536。實際設定逐 case 寫入 manifest 的 `memory`。
- `--cases ID…` 只執行指定的計畫 case；未知 ID 直接報錯。
- `--mode sweep --protein P --replica R --step FRAMES[:MEMORY] …` 依序對完整蛋白跑訓練步。同一記憶體設定在某幀數 OOM 後，跳過更多幀數；某幀數已放得下後，跳過同幀數的其他記憶體設定。被跳過的 case 標為 `skipped_by_rule` 並記錄原因。
- `--budget_minutes` 與 `--case_timeout` 明確設定 Job 內時間預算與每個 case 的逾時。

`--` 之後加上 `--training_region`（及 `--region_residues`／`--halo_rings`）時，兩種來源都改為 Training region
batch：資料集由 `MDGenDataset` 取樣，合成鏈則對第 0 幀完整分區、以 `--seed` 決定種子 group，再由同一個
`mdgen/full_length/region_batch.py` 統一欄位選取與 padding、組成 batch；只有 loss group 的 core 計入 loss（`loss_core_mask`），halo 與上下文殘基只提供上下文。#9 Ticket 3 量測一個 Training region × 250 幀即用此模式，例如
`--synthetic_residues 7500 -- … --local_spatial --training_region --num_frames 250`。

`--configs` 可列多個組態（可重複）。每個組態都使用同一份 batch、同一組初始權重
（第一個組態建立後載入其餘組態）及同一 seed（t、雜訊、dropout 相同）。

## 報告欄位

- `runs[]`：每個組態一筆
  - `precision_state`：step 內實際生效的 autocast dtype、參數 dtype、matmul 精度與 TF32 狀態。
  - `batch_shapes`：DataLoader 實際送進模型的每個張量形狀；逐筆 list（名稱、子分區 `partition`、`region` 摘要）須有 `--batch_size` 筆，子分區另列 `partition.residue_mask`；
    任一張量的 batch 維度不等於 `--batch_size` 時直接報錯（`CODING_STANDARDS.md`）。
  - `loss`、`loss_finite`、`step_time_s`（`on_train_batch_start` 到 `on_train_batch_end`，CUDA 會同步）。
  - `peak_memory_allocated_bytes`／`peak_memory_reserved_bytes`：CUDA 峰值，CPU 為 `null`。
  - `gradients`：每個參數群的 `numel`、`norm`、`max_abs`、`nonfinite`；群組為 `layers.<i>.<子模組>`、
    `ipa_layers.<i>.<子模組>` 或頂層模組（如 `t_embedder`）。梯度取自 backward 後、clipping 前。
  - `gradient_nonfinite`、`parameters_nonfinite_after_step`、`missing_gradients`
    （`--local_spatial` 下未啟用的 `cell_summaries` 參數會列在此處，屬預期）、`grad_checkpointing`。
- `comparisons[]`：後續每個組態對第一個組態
  - `gradient_cosine`、`max_abs_difference`（每個參數群）、`min_gradient_cosine` 及其群組。
  - `all_groups_pass`：所有群組 ≥ `--cosine_threshold`（預設 0.99）；任一側含 NaN／Inf 的群組 cosine 為 `null` 並判定不通過。
  - `step_time_speedup`：參考組態 step 時間 ÷ 候選組態 step 時間。單次 step 含暖機，正式加速倍數應以多次量測為準（Ticket 4）。
- `batch`（來源、名稱、各欄位形狀）、`model_args`、`seed`、`environment`
  - Training region 模式另有 `batch.regions[]`：每筆的 `loss_groups`、`halo_groups`（group 數）、
    `residues`（region 殘基數）、`loss_residues`（計入 loss 的殘基數）、`next_region_size`
    （再加下一個 loss group 時的殘基數；整個蛋白都放得下時為 `null`）。
  （主機、Python／torch／Lightning／NumPy 版本、CUDA／cuDNN、GPU 名稱、git commit、`SLURM_JOB_ID`）。

## 限制

- 合成鏈為單鏈、連續編號；多鏈 region 請用資料集來源（`<npy stem>_chain.npz`）。
- CPU 上 `bf16-train` 只驗證 autocast 與流程；TF32 及實際記憶體／速度須在 L40S 上量測（Ticket 3、4）。
- 兩次 `fp32-reference` 的 cosine 為 1、差異為 0 已在 CPU pytest 驗證；GPU 上需搭配 `--deterministic` 才保證決定性。
- 報告為訓練行為的 Precision parity 證據，不代表模型品質或 Scale validation 通過。

## L40S 小尺寸實測

2026-10-23 的 [Ticket 2 GPU 驗證](l40s-ticket2/2026-10-23/README.md) 包含 global/local 真實訓練步、FP32 重複、BF16 梯度比較、NaN/Inf 注入與跨程序重現性。僅限報告所列的小尺寸合成 batch；不代表 Training region × 250 幀或長蛋白容量驗收。
