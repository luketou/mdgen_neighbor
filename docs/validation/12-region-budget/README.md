# Training region 預算與 halo 圈數的 CPU 量測（2026-10-23）

為了決定 `--region_residues` 與 `--halo_rings` 的預設值（[ADR 0001](../../adr/0001-training-region-defaults.md)），在 CPU 上對第 0 幀分區量測每個 group 的種子 region 大小，以及不同預算下一個 Training region 能有多少 loss 殘基。這項量測只用到分區與 region 取樣，不需要 checkpoint 或 GPU。

結論：

- **預算必須 ≥ 訓練資料中最大的種子 region。** `check_region_budget` 在任何 group 放不下時會在啟動時報錯。Piezo1（6B3R）加 1 圈 halo 時，最大的種子 region 是 859 殘基，所以 `--region_residues 768` 無法啟動訓練，896 可以。
- **2 圈 halo 在 Piezo1 上需要約 1,900 殘基的預算**（最大種子 region 1,873），遠超過 FP32 在 L40S 上能訓練的大小（`docs/validation/9-49312/`：1024 殘基時 peak reserved 43.73／44.39 GiB）。
- **每個 region 的 loss 殘基差異很大**：Piezo1 在 768 預算、1 圈時，16 個 seed 的 loss 殘基為 22–367；`9-49312` 的 322 只是 seed 0 的一個 region。

## 設定

- 程式：`origin/main` @ `9dc5c40`；Python 3.11.17、torch 2.11.0+cpu（WSL，`scripts/bootstrap_cpu.sh` 建立的 CPU venv）。
- 分區參數：`spatial_cell_width` 16、`spatial_contact_radius` 16、`core_capacity` 128、`ghost_capacity` 128（ATLAS checkpoint 預設，與 `9-49312`、`12-49339` 相同）。
- 種子 region：一個 group 加上 `_halo` 的 halo group 與所有 ghost，即 `largest_seed_region` 對每個 group 取的大小。

```bash
python docs/validation/12-region-budget/synthetic_regions.py <repo> > synthetic_regions.json
python docs/validation/12-region-budget/real_protein.py <repo> 6B3R.pdb > 6B3R.txt
```

## 結構

| 名稱 | 來源 | 殘基 | group |
|---|---|---:|---:|
| Piezo1 | RCSB `6B3R.pdb`（小鼠 Piezo1 三聚體，cryo-EM 3.8 Å；6 條鏈 = 3 個主鏈 + 3 個未知片段），取每個殘基第一個 altloc 的 Cα。檔案未提交，SHA-256 `19bf6ae8…bd6f23ff` | 4,554 | 435 |
| 隨機漫步 | `train_step_probe --synthetic_residues 7500 --num_frames 250 --seed 0` 的第 0 幀 | 7,500 | 588 |
| 緊密晶格 | 5.2 Å 立方晶格的蛇行路徑（約 140 Å³／殘基，蛋白內部密度），作為最緊密的情況 | 7,500 | 294 |

隨機漫步的 seed 0 在 768 預算、1 圈時得到 767 殘基、322 個 loss 殘基，與 GPU 紀錄 49320 相同。

## 種子 region 大小（中位數／p90／最大）

| 結構 | 0 圈 | 1 圈 | 2 圈 |
|---|---|---|---|
| Piezo1 | 101／148／159 | 409／611／**859** | 843／1286／**1873** |
| 隨機漫步 | 91／154／217 | 321／565／995 | 625／1250／1842 |
| 緊密晶格 | 155／164／192 | 1038／1553／1612 | 2460／4143／5360 |

Piezo1 的 group 中，種子 region 放得進 768／1024／1536／2048 的比例：1 圈為 99%／100%／100%／100%，2 圈為 43%／71%／96%／100%。

## 每個 region 的 loss 殘基（Piezo1，seed 0–15）

| 預算 | 1 圈 | 2 圈 |
|---|---|---|
| 768 | 22–367（16 個都放得下） | 3–188（7 個 seed 放不下） |
| 1024 | 43–610（16 個都放得下） | 45–367（6 個 seed 放不下） |

合成結構的逐 seed 數值（loss、halo、上下文殘基）見 `synthetic_regions.json`；Piezo1 的完整輸出見 `6B3R.txt`。

## L40S：896 的訓練步記憶體（job 49360）

與 `9-49312` 同一設定（5 層 `--local_spatial`、`fp32-reference`、`--grad_checkpointing`、合成 7,500 殘基鏈、`--seed 0`、`--halo_rings 1`、batch 1、250 幀），只把預算改為 896。

| 項目 | 值 |
|---|---|
| Slurm job | 49360，`l40s`，node-201，GPU UUID `GPU-a78cba9e-11fb-acb6-6ff5-3e12bc8ff2eb` |
| 快照 | `/share/home/r15543056/MDgen/hpc_runs/region896-20261023T091000Z-9dc5c40/src`（`scripts/hpc_run.py`，commit `9dc5c40`、未修改） |
| 指令 | `python scripts/hpc_run.py submit --tag region896 job/submit_training_region_probe.sh --grad_checkpointing --region_residues 896 --model_dir results/workdir_896` |
| region | 38 個 loss group、18 個 halo group、869 殘基（pad 到 896）、373 個 loss 殘基；`next_region_size` 965 |
| 結果 | `loss_finite=true`、`gradient_nonfinite=0`、`parameters_nonfinite_after_step=0` |
| peak allocated／reserved | 30.23 GiB／**32.62 GiB**（容量 44.39 GiB） |
| step | 6.3 s（單次、含暖機） |

與 `9-49312` 合併，reserved 在 512／768／896／1024 為 19.46／28.06／32.62／43.73 GiB。只量一個 region、單次 step。原始資料：`probe_49360.json`、`training_region_probe_49360.out`／`.err`、`job_history`。

## 未涵蓋

- 6B3R 是靜態的 cryo-EM 模型，缺少未解析的區段，不是 MD 軌跡的第 0 幀；實際訓練資料的最大種子 region 須在資料上以 `check_region_budget` 確認。
- 記憶體只量了合成鏈的一個 region；Piezo1 的 region 組成不同，記憶體可能不同。
