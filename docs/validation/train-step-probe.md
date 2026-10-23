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

`--` 之後加上 `--training_region`（及 `--region_residues`／`--halo_rings`）時，兩種來源都改為 Training region
batch：資料集由 `MDGenDataset` 取樣，合成鏈則對第 0 幀完整分區、以 `--seed` 決定種子 group，再用同一個
`pad_region` 組成 batch；只有 loss group 的 core 計入 loss（`loss_core_mask`），halo 與上下文殘基只提供上下文。#9 Ticket 3 量測一個 Training region × 250 幀即用此模式，例如
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
