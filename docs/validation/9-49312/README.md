# L40S 量測：#9 Ticket 3（Training region 訓練步與 7,500 × 250 FP32 forward）（2026-10-23）

以 main `5b492c1`（#12 Ticket 4 的 Training region：第 0 幀完整分區挑 group、1 圈 halo、完整蛋白索引作 RoPE 位置）重新量測 #9 Ticket 3 的兩份 artifact。取代 `docs/validation/9-49243/`（#28）的訓練部分；那份用的是已移除的 #15 過渡 region。

結論：

- **FP32 裝得下訓練，但必須加 `--grad_checkpointing`。** 不加時 512 與 1024 殘基的 region 都在 forward OOM（49318、49311）。加上後 512／768／1024 殘基都完成一個 training step。
- **`--region_residues 1024` 已貼近上限**：peak reserved 43.73 GiB，GPU 容量 44.39 GiB。768 為 28.06 GiB，留有約 16 GiB 餘裕。建議把預設改為 768，或保留 1024 但只在確認不同 region 不會超過時使用（本次每個大小只量一個 region）。
- **完整 7,500 × 250 的 FP32 生成 forward 不 OOM**（49316）：peak allocated 30.04 GiB、reserved 32.77 GiB，單次 forward 中位數 16.98 s。

## 執行環境

| 項目 | 值 |
|---|---|
| Slurm job | 49311、49312、49318、49319、49320（訓練步）；49316（forward）。分區 `l40s`，node-201 |
| GPU | NVIDIA L40S，46068 MiB（PyTorch 回報 44.39 GiB）。49312、49318、49319、49320 在 UUID `GPU-83a6811b-8c82-590e-4fdb-e85d9e786e61`；49311、49316 在 `GPU-a78cba9e-11fb-acb6-6ff5-3e12bc8ff2eb` |
| 程式版本 | main @ `5b492c1`（#30 合併後） |
| 軟體 | Python 3.11.12、torch 2.11.0+cu128、Lightning 2.6.5、CUDA 12.8、cuDNN 91900 |
| 精度 | `fp32-reference`：Lightning `32-true`、autocast 關閉、參數 float32、matmul `highest`、TF32 關閉（forward 同為 `highest`、TF32 關閉） |
| 模型 | 5 層 `--local_spatial`、`--sim_condition --prepend_ipa --num_frames 250 --batch_size 1`；`spatial_cell_width` 16、`spatial_contact_radius` 16、`core_capacity` 128、`ghost_capacity` 128 |

快照：`/share/home/r15543056/MDgen/hpc_runs/l40s-20261023T071407Z-5b492c1/src`，由 `git archive HEAD` 產生（不含 `tests/reference`），SHA-256 驗證後設為唯讀；`src/results`、`src/logs` 為指向 run 目錄的 symlink。`probe.json` 的 `git_commit` 為 `null`，因為快照沒有 `.git`。

提交指令（在 `<run>/src`，`MDGEN_VENV=/share/home/r15543056/.venvs/mdgen`）：

```bash
sbatch --chdir=<run>/src job/submit_training_region_probe.sh --model_dir <run>/workdir_<tag> --region_residues N [--grad_checkpointing]
MODEL_DIR=<run>/workdir_fwd7500 sbatch --chdir=<run>/src ... --wrap "source job/env.sh; python scripts/diagnostics/scale_sdpa.py \
  --mode local --length 7500 --operation forward --ffn-policy chunked --ffn-chunk-size 65536 --preprocessing native --outdir results/fwd7500_\$SLURM_JOB_ID"
```

## 訓練步（`mdgen.train_step_probe`，合成 7,500 殘基單鏈，`--seed 0`、`--halo_rings 1`）

| Job | `--region_residues` | checkpointing | region（loss／halo group，實際殘基，loss 殘基） | 結果 | peak allocated | peak reserved | step |
|---|---:|---|---|---|---:|---:|---:|
| 49318 | 512 | 無 | — | **OOM** | —（OOM 時已配置 42.52 GiB） | — | — |
| 49311 | 1024 | 無 | — | **OOM** | —（OOM 時已配置 43.25 GiB） | — | — |
| 49319 | 512 | `--grad_checkpointing` | 25／10，488，215 | 完成 | 17.26 GiB | 19.46 GiB | 3.58 s |
| 49320 | 768 | `--grad_checkpointing` | 33／15，767，322 | 完成 | 26.18 GiB | 28.06 GiB | 5.40 s |
| 49312 | 1024 | `--grad_checkpointing` | 49／29，1024，484 | 完成 | 39.28 GiB | **43.73 GiB** | 11.20 s |

三個完成的 job 都是 `loss_finite=true`、`gradient_nonfinite=0`、`parameters_nonfinite_after_step=0`。

- OOM 位置：49311 在 local spatial attention 的 RoPE（`mdgen/full_length/attention.py:44` `rotate`），49318 在 `mdgen/model/layers.py:15` `modulate`；兩者都在 forward。
- step 時間是單次、含暖機，只能看量級。1024 的 11.2 s 明顯高於由 512→768 外推的值，可能與 allocator 接近容量有關，本次沒有量測依據，**未解釋**。
- `region_residues` 是上限，實際殘基數由取樣決定（`next_region_size` 為再加一個 group 會到的大小）。每種大小只量一個 region（同一 seed）；其他 region 的殘基數與 ghost 數不同，記憶體也會不同。
- 合成鏈每幀為獨立隨機漫步，loss 數值沒有意義，不用於任何結論。

## 完整 7,500 × 250 FP32 forward（`scripts/diagnostics/scale_sdpa.py`）

| 項目 | 49316 |
|---|---|
| 輸入 | `5e3e_A` 重複排在 3-D 格點上、截成 7,500 殘基（73 條鏈），250 幀，batch 1 |
| 模型 | ATLAS checkpoint（SHA-256 `bc90c66e…`），`local_spatial`，`cell_summaries` 關閉，FFN chunk 65,536 |
| 結果 | `status=success`、`input_all_finite=true`、`output_all_finite=true` |
| peak allocated／reserved | 30.04 GiB／32.77 GiB |
| forward 時間 | 5 次：16.90–16.99 s，中位數 16.98 s |

這是資源量測，不是對真實 7,500 殘基蛋白的品質宣稱。只跑 forward，沒有跑完整 ODE 生成；完整生成的 wall-clock 與 NFE 屬 Ticket 8。

## 原始資料

| 檔案 | 內容 |
|---|---|
| `probe_49312.json`、`probe_49319.json`、`probe_49320.json` | 訓練步完整報告（遠端 `<run>/results/training_region_probe_<job>/probe.json`） |
| `training_region_probe_<job>.out`／`.err` | 五個訓練步 job 的 log；49311、49318 的 `.err` 含 OOM traceback |
| `fwd7500_49316.json`、`.out`／`.err` | forward 報告與 log（遠端 `<run>/results/fwd7500_49316/local_7500_forward.json`） |
| `job_history`、`meta.json` | 本快照的所有提交紀錄與快照來源 |

## 同一快照中沒有產生結果的 job

| Job | 原因 |
|---|---|
| 49304–49307 | `module load` 的 `auto_gpu` 在 `bash -c` 中載入 `~/.bashrc`（繼承了 SSH 的 `SSH_CLIENT`），`.bashrc` 在 compute node 建立 `/run/user/<uid>` 失敗。改為提交前 `unset SSH_CLIENT SSH_CONNECTION SSH_TTY` 並設 `XDG_RUNTIME_DIR=/tmp/xdg-$USER` 後重送 |
| 49313、49314 | `mdgen/logger.py` 在唯讀快照建立 `./workdir` 失敗；改設 `MODEL_DIR=<run>/workdir_*` 後重送為 49316、49317 |

## 未涵蓋

- 只量 `fp32-reference`；BF16 的梯度 cosine 與加速倍數屬 Ticket 4。
- batch size 1、單一 region、單次 step；沒有量多個 region 的記憶體分布。
