# L40S 量測：#9 Ticket 3 訓練部分（2026-10-23）

以 `job/submit_training_region_probe.sh` 量測 5 層 `--local_spatial` ATLAS 組態，在一個 Training region × 250 幀上跑一次 `fp32-reference` training step。

結論：**FP32 不使用 checkpointing 時裝不下**，job 49241 在 forward 中 OOM。**加上 `--grad_checkpointing` 後裝得下**，job 49243 的峰值 allocated 為 19.01 GiB，GPU 容量 44.39 GiB。以上只涵蓋本次量到的單一 region（center 6044、core 256、ghost 137），以及 batch size 1。

## 執行環境

| 項目 | 值 |
|---|---|
| Slurm job | 49241（無 checkpointing）、49243（`--grad_checkpointing`），分區 `l40s`（node-201） |
| GPU | NVIDIA L40S，UUID `GPU-a78cba9e-11fb-acb6-6ff5-3e12bc8ff2eb`，46068 MiB（PyTorch 回報 44.39 GiB） |
| 程式版本 | `claude/focused-noether-qqukbh` @ `58d7907`（#21） |
| 軟體 | Python 3.11.12、torch 2.11.0+cu128、Lightning 2.6.5、CUDA 12.8、cuDNN 91900 |
| 精度 | `fp32-reference`：Lightning `32-true`、autocast 關閉、參數 float32、matmul `highest`、TF32 關閉 |
| 輸入 | `--synthetic_residues 7500 --seed 0`，合成單鏈 |

兩個 job 都用同一份程式快照：`/share/home/r15543056/MDgen/hpc_runs/probe-20261022T165754Z-58d7907/src`。快照由 `git archive` 產生，經 SHA-256 驗證，且為唯讀。`probe.json` 的 `environment.git_commit` 是 `null`，因為快照裡沒有 `.git`；commit 以快照來源為準。

提交指令：

```bash
cd <run>/src
MDGEN_VENV=/share/home/r15543056/.venvs/mdgen sbatch --chdir=<run>/src \
  job/submit_training_region_probe.sh --model_dir <run>/workdir
MDGEN_VENV=/share/home/r15543056/.venvs/mdgen sbatch --chdir=<run>/src \
  job/submit_training_region_probe.sh --model_dir <run>/workdir --grad_checkpointing
```

`--model_dir` 只決定 `mdgen/logger.py` 的 log 寫到哪裡。`parse_train_args` 預設寫到相對路徑 `workdir/`，在唯讀快照中無法建立（job 49238 因此失敗），所以改指到 run 目錄。這個參數不影響模型組態。

## 原始資料

| 檔案 | 內容 |
|---|---|
| `probe.json` | job 49243 的完整報告（遠端：`<run>/results/training_region_probe_49243/probe.json`） |
| `training_region_probe_49243.out`／`.err` | job 49243 的 log |
| `training_region_probe_49241.out`／`.err` | job 49241 的 log，含 OOM traceback；這個 job 沒有產生 `probe.json` |

## Region

來自 job 49243 的 `batch.regions[0]`：

| 欄位 | 值 |
|---|---:|
| `center` | 6044 |
| `core_count` | 256 |
| `ghost_count` | 137（上限 `--region_ghost 256`） |
| `mandatory_neighbor_count`／`mandatory_retained_count` | 31／31 |
| `spatial_neighbor_count`／`spatial_retained_count` | 137／137 |
| `spatial_truncated_count`（被捨棄的空間鄰居數） | **0** |

模型輸入形狀為 `trans` `[1, 250, 512, 3]`：每幀 512 個位置，其中 393 個是實際殘基（core 256 + ghost 137）。ghost 為什麼少於上限 256，本次沒有量測依據，**未解釋**。

job 49241 用的是同樣的 `--seed 0` 與合成輸入，但它在寫出報告前就 OOM 了，所以沒有自己的 region 紀錄，無法逐欄確認兩個 job 的 region 相同。

## 結果

| 項目 | 49241（無 checkpointing） | 49243（`--grad_checkpointing`） |
|---|---|---|
| 結果 | **OOM**（`torch.OutOfMemoryError`） | 完成 1 個 training step |
| 峰值 allocated | 未記錄；OOM 當下 PyTorch 已配置 42.96 GiB | **19.01 GiB**（20,408,259,584 bytes） |
| 峰值 reserved | 未記錄；OOM 當下行程共使用 44.19 GiB | **20.63 GiB**（22,154,313,728 bytes） |
| step 時間 | — | 4.93 s（單次，含暖機） |
| `loss_finite` | — | `true` |
| `gradient_nonfinite` | — | 0 |
| `parameters_nonfinite_after_step` | — | 0 |

job 49241 的 OOM 訊息：嘗試再配置 212.00 MiB 時，GPU 只剩 197.31 MiB 可用；另有 723.26 MiB 是 reserved 但未配置。發生位置是 forward 中的 local spatial attention（`mdgen/full_length/attention.py:44`，`rotate`）。traceback 沒有顯示是第幾層。

`missing_gradients` 列出了每層的 `cell_summaries` 參數。依 `docs/validation/train-step-probe.md`，這是 `--local_spatial` 下未啟用的模組，屬於預期現象。

合成鏈的 loss 數值沒有意義，本紀錄不使用它。

## 範圍與未涵蓋

- 只量了一個 region、一個 seed、batch size 1、單次 step。其他 region 大小、其他 center，或 core/ghost 都填滿 256 的情況，本次沒有量。
- 4.93 s 是包含暖機的單次 step，不能當作穩定的 step 時間。
- 沒有量 bf16 或其他精度組態。
- Ticket 3 的另一份 artifact（完整 7,500 × 250 的 FP32 生成不 OOM）不在本紀錄範圍內。

## 同一快照的其他 job

這些 job 沒有產生量測結果，列出來供追溯：

| Job | 結果 |
|---|---|
| 49072、49076 | node-201 GPU 故障（`/dev/nvidia*` 回 EIO、`nvidia-smi` 印 `No devices were found`），探針沒有執行 |
| 49079 | 診斷：兩張 GPU 都回 EIO |
| 49236 | 管理員修復後的診斷：兩張 GPU 都可開啟，`nvidia-smi` 與 torch matmul 正常 |
| 49238 | 探針因 `PermissionError: 'workdir'` 失敗（唯讀快照），改加 `--model_dir` 後重送為 49241 |
